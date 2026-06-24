#!/usr/bin/env python3
"""
Piper TTS API – GPU via binário compilado
Arquitetura: pool de processos Piper → síntese paralela (threads) → mixagem em processos separados
"""

import os
import re
import io
import json
import time
import queue
import logging
import subprocess
import threading
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

# =============================================================================
# Configuração de logging MUITO detalhada
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("piper-api")

# =============================================================================
# Função para executar comandos e capturar saída (usada no diagnóstico)
# =============================================================================
def run_cmd(cmd: list, timeout: int = 10) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.stdout + res.stderr
    except Exception as e:
        return f"Erro ao executar {cmd}: {e}"

# =============================================================================
# Diagnóstico da GPU e ambiente CUDA (logo no início)
# =============================================================================
logger.info("=" * 70)
logger.info("DIAGNÓSTICO DE AMBIENTE")
logger.info("=" * 70)

# 1. nvidia-smi
nvidia_output = run_cmd(["nvidia-smi"], timeout=5)
logger.info("nvidia-smi:\n" + nvidia_output)

# 2. Variáveis CUDA
cuda_home = os.environ.get("CUDA_HOME", "não definida")
ld_library = os.environ.get("LD_LIBRARY_PATH", "não definida")
logger.info(f"CUDA_HOME={cuda_home}")
logger.info(f"LD_LIBRARY_PATH={ld_library}")

# 3. Verificar se o binário piper existe e é executável
piper_bin = "/app/piper-bin"
if os.path.isfile(piper_bin):
    logger.info(f"Binário piper encontrado em {piper_bin}")
    # Testar execução do piper --help para ver se as libs são carregadas
    help_output = run_cmd([piper_bin, "--help"], timeout=5)
    logger.info("piper --help:\n" + help_output[:1000])
else:
    logger.critical(f"Binário piper NÃO encontrado em {piper_bin}!")

# 4. Verificar se há dispositivos CUDA via Python (se onnxruntime-gpu estiver)
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    logger.info(f"ONNX Runtime disponível. Providers: {providers}")
except ImportError:
    logger.info("onnxruntime Python não instalado (não obrigatório, pois usamos binário)")

logger.info("=" * 70)

# =============================================================================
# Diretórios
# =============================================================================
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

for d in (VOICES_DIR, AMBIENT_DIR, EFFECTS_DIR):
    d.mkdir(exist_ok=True)

# =============================================================================
# Paralelismo – Ajustável via variáveis de ambiente
# =============================================================================
SYNTHESIS_THREADS = int(os.getenv("SYNTHESIS_THREADS", "8"))
MIXING_PROCESSES = int(os.getenv("MIXING_PROCESSES", "8"))
PIPER_POOL_SIZE = int(os.getenv("PIPER_POOL_SIZE", "2"))

logger.info(f"Configuração: SYNTHESIS_THREADS={SYNTHESIS_THREADS}, "
            f"MIXING_PROCESSES={MIXING_PROCESSES}, PIPER_POOL_SIZE={PIPER_POOL_SIZE}")

# =============================================================================
# Executores globais
# =============================================================================
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

synthesis_executor = ThreadPoolExecutor(max_workers=SYNTHESIS_THREADS)
mixing_executor = ProcessPoolExecutor(max_workers=MIXING_PROCESSES)

logger.info("Executores de síntese e mixagem iniciados.")

# =============================================================================
# PiperProcess – processo individual do binário piper
# =============================================================================
class PiperProcess:
    def __init__(self, model_path: str, config_path: str, use_cuda: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.use_cuda = use_cuda
        self.process = None
        self.lock = threading.Lock()
        self._stderr_lines = []
        self._stderr_thread = None
        self._start()

    def _start(self):
        cmd = [
            "/app/piper-bin",
            "--model", self.model_path,
            "--config", self.config_path,
            "--json-input",
            "--output-raw"
        ]
        if self.use_cuda:
            cmd.append("--cuda")

        logger.info(f"[PIPER] Iniciando: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
        except Exception as e:
            logger.error(f"[PIPER] Falha ao criar subprocesso: {e}")
            raise

        self._stderr_lines = []
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        # Aguarda um pouco e verifica se o processo ainda está vivo
        time.sleep(2.0)
        if self.process.poll() is not None:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Processo morreu na inicialização. stderr: {stderr_tail[-500:]}")
            raise RuntimeError(f"Piper morreu ao iniciar. stderr: {stderr_tail[-200:]}")

        logger.info(f"[PIPER] Pronto (PID {self.process.pid}) para modelo {Path(self.model_path).name}")

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, b''):
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 100:
                self._stderr_lines.pop(0)

    def _get_stderr_tail(self) -> str:
        return b"".join(self._stderr_lines).decode(errors='replace')

    def synthesize(self, text: str, length_scale: float = 1.0,
                   noise_scale: float = 0.667, noise_w_scale: float = 0.8) -> Tuple[bytes, int]:
        request = {
            "text": text,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w": noise_w_scale
        }
        with self.lock:
            try:
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                self.process.stdin.flush()

                line = self.process.stdout.readline()
                if not line:
                    stderr_tail = self._get_stderr_tail()
                    logger.error(f"[PIPER] Processo morreu. stderr: {stderr_tail[-500:]}")
                    raise RuntimeError(f"Processo piper morreu (stdout vazio). stderr: {stderr_tail[-200:]}")

                response = json.loads(line)
                num_samples = response.get("num_samples", 0)
                sample_rate = response.get("sample_rate", 22050)
                raw_audio = self.process.stdout.read(num_samples * 2)

                if len(raw_audio) != num_samples * 2:
                    logger.warning(f"[PIPER] Tamanho de áudio inesperado: esperado {num_samples*2}, recebido {len(raw_audio)}")

                return raw_audio, sample_rate

            except (json.JSONDecodeError, KeyError) as e:
                stderr_tail = self._get_stderr_tail()
                logger.error(f"[PIPER] Erro de parsing: {e}. stderr: {stderr_tail[-500:]}")
                self._restart()
                raise RuntimeError(f"Falha na comunicação com piper: {e}")
            except (BrokenPipeError, Exception) as e:
                stderr_tail = self._get_stderr_tail()
                logger.error(f"[PIPER] Exceção: {e}. stderr: {stderr_tail[-500:]}")
                self._restart()
                raise

    def _restart(self):
        logger.warning("[PIPER] Reiniciando processo...")
        if self.process:
            try:
                self.process.kill()
            except:
                pass
            self.process = None
        self._start()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

# =============================================================================
# Pool de processos Piper por voz
# =============================================================================
class PiperProcessPool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = PIPER_POOL_SIZE):
        self.pool = queue.Queue(maxsize=pool_size)
        for i in range(pool_size):
            try:
                proc = PiperProcess(model_path, config_path, use_cuda=True)
                self.pool.put(proc)
                logger.info(f"  → Piper worker {i+1}/{pool_size} para {Path(model_path).stem}")
            except Exception as e:
                logger.error(f"  ✗ Falha ao criar worker Piper {i+1}: {e}")
        if self.pool.empty():
            raise RuntimeError("Nenhum processo Piper pôde ser iniciado.")

    def synthesize(self, text, length_scale, noise_scale, noise_w_scale):
        proc = self.pool.get()
        try:
            return proc.synthesize(text, length_scale, noise_scale, noise_w_scale)
        finally:
            self.pool.put(proc)

# =============================================================================
# Carregamento de vozes
# =============================================================================
voices_registry: Dict[str, PiperProcessPool] = {}

def load_voice_from_folder(voice_name: str, voice_path: Path) -> PiperProcessPool:
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_path}")
    model_path = str(onnx_files[0])

    json_path = voice_path / f"{onnx_files[0].stem}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum .json para {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    logger.info(f"Carregando voz {voice_name} (pool size={PIPER_POOL_SIZE})")
    pool = PiperProcessPool(model_path, config_path, pool_size=PIPER_POOL_SIZE)
    logger.info(f"✅ Voz {voice_name} pronta (GPU)")
    return pool

for item in VOICES_DIR.iterdir():
    if item.is_dir():
        voice_name = item.name
        try:
            voices_registry[voice_name] = load_voice_from_folder(voice_name, item)
        except Exception as e:
            logger.error(f"❌ Voz {voice_name}: {e}")

for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voices_registry:
        continue
    json_file = onnx_file.with_suffix(".onnx.json")
    if json_file.exists():
        try:
            voices_registry[voice_name] = load_voice_from_folder(voice_name, VOICES_DIR)
        except Exception as e:
            logger.error(f"❌ Voz raiz {voice_name}: {e}")

logger.info(f"Total de vozes carregadas: {len(voices_registry)}")
MODEL_LOADED = len(voices_registry) > 0

# =============================================================================
# WARM-UP: Teste completo do pipeline (voz faber + efeito tosse + ambiente ubs)
# =============================================================================
logger.info("=" * 70)
logger.info("🔥 INICIANDO WARM-UP (teste de funcionamento)")
logger.info("=" * 70)

warmup_success = False
if "faber" in voices_registry:
    logger.info("Voz 'faber' encontrada. Preparando warm-up...")

    # Frase de teste
    warmup_text = "Teste de aquecimento [tosse] finalizado."
    warmup_effects = {"[tosse]": "tosse.wav"}
    warmup_ambient_file = "ubs.wav"
    warmup_ambient_vol = -15.0

    # Verificar se arquivos de efeito/ambiente existem
    effect_path = EFFECTS_DIR / "tosse.wav"
    ambient_path = AMBIENT_DIR / "ubs.wav"
    if not effect_path.exists():
        logger.warning(f"Arquivo de efeito tosse.wav não encontrado em {EFFECTS_DIR}")
    if not ambient_path.exists():
        logger.warning(f"Arquivo de ambiente ubs.wav não encontrado em {AMBIENT_DIR}")

    try:
        # Executa uma síntese simples usando a voz faber
        logger.info("Sintetizando frase de warm-up...")
        pool = voices_registry["faber"]

        # Simula o processamento rápido (modo simples)
        # 1. Síntese da fala
        pcm, sr = pool.synthesize("Teste de aquecimento", length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)
        logger.info(f"Síntese de fala OK: {len(pcm)} bytes, {sr} Hz")

        # 2. Carregar efeito (apenas para verificar)
        if effect_path.exists():
            with open(effect_path, "rb") as f:
                effect_bytes = f.read()
            logger.info(f"Efeito carregado: {len(effect_bytes)} bytes")
        else:
            effect_bytes = None

        # 3. Carregar ambiente
        if ambient_path.exists():
            with open(ambient_path, "rb") as f:
                ambient_bytes = f.read()
            logger.info(f"Ambiente carregado: {len(ambient_bytes)} bytes")
        else:
            ambient_bytes = None

        # 4. Mixagem mínima (apenas para validar o processo)
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
        if sr != 22050:
            seg = seg.set_frame_rate(22050)
        combined = seg  # sem efeito/ambiente para simplificar, mas podemos incluir
        buf = io.BytesIO()
        combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
        webm_bytes = buf.getvalue()
        logger.info(f"Warm-up concluído com sucesso! WebM gerado: {len(webm_bytes)} bytes")
        warmup_success = True
    except Exception as e:
        logger.critical(f"❌ WARM-UP FALHOU: {e}")
        # Não levanta exceção para não impedir a inicialização, mas loga crítico
else:
    logger.warning("Voz 'faber' não encontrada. Pule warm-up (adicione a voz ou ajuste o nome).")
    logger.info("Para testar, copie a voz desejada para /app/voices/faber/")

logger.info(f"Warm-up {'bem-sucedido' if warmup_success else 'FALHOU'}")
logger.info("=" * 70)

# =============================================================================
# Função de mixagem (executada em processo separado)
# =============================================================================
def mixing_task(
    ordered_items: list,
    ambient_bytes: Optional[bytes],
    ambient_volume_db: float,
    target_dbfs: float = -20.0
) -> bytes:
    log = logging.getLogger("mixing")
    log.info(f"Iniciando mixagem de {len(ordered_items)} segmentos")
    audio_chunks = []

    for i, item in enumerate(ordered_items):
        if item['type'] == 'speech':
            seg = AudioSegment(
                data=item['pcm'],
                sample_width=2,
                frame_rate=item['sample_rate'],
                channels=1
            )
            if item['sample_rate'] != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
            log.debug(f"Segmento de fala {i}: {len(seg)/1000:.2f}s")

        elif item['type'] == 'effect':
            seg = AudioSegment.from_wav(io.BytesIO(item['wav_bytes']))
            if seg.frame_rate != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
            log.debug(f"Efeito {i}: {len(seg)/1000:.2f}s")

    if not audio_chunks:
        raise ValueError("Nenhum áudio para mixar")

    combined = sum(audio_chunks, AudioSegment.empty())
    log.info(f"Áudio combinado: {len(combined)/1000:.2f}s, dBFS={combined.dBFS:.1f}")

    if combined.dBFS != target_dbfs:
        gain = target_dbfs - combined.dBFS
        log.info(f"Aplicando ganho de {gain:.1f} dB")
        combined = combined.apply_gain(gain)

    if ambient_bytes:
        log.info("Mixando ambiente")
        ambient = AudioSegment.from_wav(io.BytesIO(ambient_bytes))
        if ambient.frame_rate != combined.frame_rate:
            ambient = ambient.set_frame_rate(combined.frame_rate)
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)

    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
    log.info(f"Mixagem concluída: {buf.tell()} bytes")
    return buf.getvalue()

# =============================================================================
# Modelos Pydantic
# =============================================================================
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class TTSRequest(BaseModel):
    voice: Optional[str] = Field(None, description="Nome da voz (modo único)")
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="Piper TTS API GPU (pipeline otimizado)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio_geral = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...' | efeitos={list(req.effects.keys())} | ambient={req.ambient.enabled}")

    # ---------- 1. Parse do texto e mapeamento de falantes ----------
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório no modo simples")
        if req.voice not in voices_registry:
            raise HTTPException(404, f"Voz '{req.voice}' não encontrada")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, ns, nw)
        for role, (v, _, _, _) in speaker_map.items():
            if v not in voices_registry:
                raise HTTPException(404, f"Voz '{v}' do speaker '{role}' não encontrada")
        current_role = None

    # ---------- 2. Dividir texto e classificar partes ----------
    parts = re.split(r'(\[.*?\])', req.text)
    logger.info(f"Texto dividido em {len(parts)} partes")

    ordered_items = []
    synthesis_tasks = []
    effect_preload = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
                logger.info(f"  → Speaker: {role} ({speaker_map[role][0]})")
            continue

        if part in req.effects:
            effect_name = req.effects[part]
            logger.info(f"  → Efeito: {part} -> {effect_name}")
            if effect_name not in effect_preload:
                voice_name_for_effect = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                voice_dir = Path(voices_registry[voice_name_for_effect].model_path).parent if voice_name_for_effect in voices_registry else None
                effect_path = None
                if voice_dir:
                    candidate = voice_dir / effect_name
                    if candidate.exists():
                        effect_path = candidate
                if not effect_path:
                    candidate = EFFECTS_DIR / effect_name
                    if candidate.exists():
                        effect_path = candidate
                if not effect_path:
                    raise HTTPException(404, f"Efeito '{effect_name}' não encontrado")
                with open(effect_path, "rb") as f:
                    effect_preload[effect_name] = f.read()
                logger.debug(f"Efeito carregado: {len(effect_preload[effect_name])} bytes")
            ordered_items.append({'type': 'effect', 'wav_bytes': effect_preload[effect_name]})
            continue

        # Texto para síntese
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto. Use [papel].")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        idx = len(ordered_items)
        ordered_items.append({'type': 'speech', 'pcm': None, 'sample_rate': None})
        synthesis_tasks.append((idx, voice_name, part, speed, noise_s, noise_w))

    if not synthesis_tasks and not any(item['type'] == 'speech' for item in ordered_items):
        raise HTTPException(400, "Nenhum texto para sintetizar")

    # ---------- 3. Disparar sínteses em paralelo ----------
    logger.info(f"Disparando {len(synthesis_tasks)} sínteses em paralelo")
    futures = {}
    for idx, voice_name, text, speed, noise_s, noise_w in synthesis_tasks:
        pool = voices_registry[voice_name]
        fut = synthesis_executor.submit(pool.synthesize, text, speed, noise_s, noise_w)
        futures[fut] = idx

    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            pcm, sr = fut.result()
            ordered_items[idx]['pcm'] = pcm
            ordered_items[idx]['sample_rate'] = sr
            logger.debug(f"Síntese {idx} concluída ({len(pcm)} bytes, {sr} Hz)")
        except Exception as e:
            logger.error(f"Síntese {idx} falhou: {e}")
            raise HTTPException(500, f"Erro na síntese de voz: {e}")

    tempo_sintese = time.perf_counter() - inicio_geral
    logger.info(f"Sínteses concluídas em {tempo_sintese:.2f}s")

    # ---------- 4. Preparar ambiente ----------
    ambient_bytes = None
    if req.ambient.enabled and req.ambient.file:
        ambient_path = AMBIENT_DIR / f"{req.ambient.file}.wav"
        if not ambient_path.exists():
            raise HTTPException(404, f"Ambiente '{req.ambient.file}.wav' não encontrado")
        with open(ambient_path, "rb") as f:
            ambient_bytes = f.read()
        logger.info(f"Ambiente carregado: {len(ambient_bytes)} bytes")

    # ---------- 5. Mixagem em processo separado ----------
    logger.info(f"Enviando mixagem para pool de {MIXING_PROCESSES} processos")
    loop = asyncio.get_event_loop()
    try:
        webm_bytes = await loop.run_in_executor(
            mixing_executor,
            mixing_task,
            ordered_items,
            ambient_bytes,
            req.ambient.volume_db,
            -20.0
        )
    except Exception as e:
        logger.error(f"Erro na mixagem: {e}")
        raise HTTPException(500, f"Erro na mixagem: {e}")

    tempo_total = time.perf_counter() - inicio_geral
    duracao_estimada = len(webm_bytes) / 8000  # estimativa grosseira
    logger.info(f"✅ Requisição finalizada | total={tempo_total:.2f}s | WebM={len(webm_bytes)} bytes | RTF≈{tempo_total/duracao_estimada:.3f}")
    return Response(content=webm_bytes, media_type="audio/webm")

# ---------- Health checks ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if MODEL_LOADED:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    gpu_ok = all(
        entry.is_alive() if hasattr(entry, "is_alive") else True
        for entry in voices_registry.values()
    ) if voices_registry else False
    return {
        "status": "ok",
        "gpu": gpu_ok,
        "warmup": warmup_success,
        "voices_loaded": list(voices_registry.keys()),
        "total_voices": len(voices_registry)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)#!/usr/bin/env python3
"""
Piper TTS API – GPU via binário compilado
Arquitetura: pool de processos Piper → síntese paralela (threads) → mixagem em processos separados
"""

import os
import re
import io
import json
import time
import queue
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

# =============================================================================
# Configuração de logging MUITO detalhada
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("piper-api")

# =============================================================================
# Diretórios
# =============================================================================
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

for d in (VOICES_DIR, AMBIENT_DIR, EFFECTS_DIR):
    d.mkdir(exist_ok=True)

# =============================================================================
# Paralelismo – Ajustável via variáveis de ambiente
# =============================================================================
SYNTHESIS_THREADS = int(os.getenv("SYNTHESIS_THREADS", "8"))      # threads para enviar sínteses
MIXING_PROCESSES = int(os.getenv("MIXING_PROCESSES", "8"))        # processos de mixagem
PIPER_POOL_SIZE = int(os.getenv("PIPER_POOL_SIZE", "2"))          # processos piper por voz

logger.info(f"Configuração: SYNTHESIS_THREADS={SYNTHESIS_THREADS}, "
            f"MIXING_PROCESSES={MIXING_PROCESSES}, PIPER_POOL_SIZE={PIPER_POOL_SIZE}")

# =============================================================================
# Executores globais (criados uma vez na inicialização)
# =============================================================================
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

synthesis_executor = ThreadPoolExecutor(max_workers=SYNTHESIS_THREADS)
mixing_executor = ProcessPoolExecutor(max_workers=MIXING_PROCESSES)

logger.info("Executores de síntese (threads) e mixagem (processos) iniciados.")

# =============================================================================
# PiperProcess – processo individual do binário piper
# =============================================================================
class PiperProcess:
    def __init__(self, model_path: str, config_path: str, use_cuda: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.use_cuda = use_cuda
        self.process = None
        self.lock = threading.Lock()
        self._stderr_lines = []
        self._stderr_thread = None
        self._start()

    def _start(self):
        cmd = [
            "/app/piper-bin",
            "--model", self.model_path,
            "--config", self.config_path,
            "--json-input",
            "--output-raw"
        ]
        if self.use_cuda:
            cmd.append("--cuda")

        logger.info(f"[PIPER] Iniciando: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
        except Exception as e:
            logger.error(f"[PIPER] Falha ao criar subprocesso: {e}")
            raise

        # Thread para capturar stderr continuamente
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        # Aguarda um pouco e verifica se o processo ainda está vivo
        time.sleep(2.0)
        if self.process.poll() is not None:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Processo morreu na inicialização. stderr: {stderr_tail[-500:]}")
            raise RuntimeError(f"Piper morreu ao iniciar. stderr: {stderr_tail[-200:]}")

        logger.info(f"[PIPER] Pronto (PID {self.process.pid}) para modelo {Path(self.model_path).name}")

    def _read_stderr(self):
        """Lê stderr continuamente e guarda as últimas 100 linhas."""
        for line in iter(self.process.stderr.readline, b''):
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 100:
                self._stderr_lines.pop(0)

    def _get_stderr_tail(self) -> str:
        return b"".join(self._stderr_lines).decode(errors='replace')

    def synthesize(self, text: str, length_scale: float = 1.0,
                   noise_scale: float = 0.667, noise_w_scale: float = 0.8) -> Tuple[bytes, int]:
        """Retorna (pcm_bytes, sample_rate)."""
        request = {
            "text": text,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w": noise_w_scale
        }
        with self.lock:
            try:
                logger.debug(f"[PIPER] Enviando: {json.dumps(request)}")
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                self.process.stdin.flush()

                line = self.process.stdout.readline()
                if not line:
                    stderr_tail = self._get_stderr_tail()
                    logger.error(f"[PIPER] Processo morreu. stderr: {stderr_tail[-500:]}")
                    raise RuntimeError(f"Processo piper morreu (stdout vazio). stderr: {stderr_tail[-200:]}")

                response = json.loads(line)
                logger.debug(f"[PIPER] Resposta JSON: {response}")

                num_samples = response.get("num_samples", 0)
                sample_rate = response.get("sample_rate", 22050)
                raw_audio = self.process.stdout.read(num_samples * 2)

                if len(raw_audio) != num_samples * 2:
                    logger.warning(f"[PIPER] Tamanho de áudio inesperado: esperado {num_samples*2}, recebido {len(raw_audio)}")

                logger.debug(f"[PIPER] Síntese OK: {num_samples} amostras, {sample_rate} Hz")
                return raw_audio, sample_rate

            except (json.JSONDecodeError, KeyError) as e:
                stderr_tail = self._get_stderr_tail()
                logger.error(f"[PIPER] Erro de parsing: {e}. stderr: {stderr_tail[-500:]}")
                self._restart()
                raise RuntimeError(f"Falha na comunicação com piper: {e}")
            except (BrokenPipeError, Exception) as e:
                stderr_tail = self._get_stderr_tail()
                logger.error(f"[PIPER] Exceção: {e}. stderr: {stderr_tail[-500:]}")
                self._restart()
                raise

    def _restart(self):
        logger.warning("[PIPER] Reiniciando processo...")
        if self.process:
            try:
                self.process.kill()
            except:
                pass
            self.process = None
        self._start()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

# =============================================================================
# Pool de processos Piper por voz
# =============================================================================
class PiperProcessPool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = PIPER_POOL_SIZE):
        self.pool = queue.Queue(maxsize=pool_size)
        for i in range(pool_size):
            try:
                proc = PiperProcess(model_path, config_path, use_cuda=True)
                self.pool.put(proc)
                logger.info(f"  → Piper worker {i+1}/{pool_size} para {Path(model_path).stem}")
            except Exception as e:
                logger.error(f"  ✗ Falha ao criar worker Piper {i+1}: {e}")
        if self.pool.empty():
            raise RuntimeError("Nenhum processo Piper pôde ser iniciado.")

    def synthesize(self, text, length_scale, noise_scale, noise_w_scale):
        proc = self.pool.get()
        try:
            return proc.synthesize(text, length_scale, noise_scale, noise_w_scale)
        finally:
            self.pool.put(proc)

# =============================================================================
# Registro de vozes
# =============================================================================
voices_registry: Dict[str, PiperProcessPool] = {}

def load_voice_from_folder(voice_name: str, voice_path: Path) -> PiperProcessPool:
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_path}")
    model_path = str(onnx_files[0])

    json_path = voice_path / f"{onnx_files[0].stem}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum .json para {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    logger.info(f"Carregando voz {voice_name} (pool size={PIPER_POOL_SIZE})")
    pool = PiperProcessPool(model_path, config_path, pool_size=PIPER_POOL_SIZE)
    logger.info(f"✅ Voz {voice_name} pronta (GPU)")
    return pool

# Carrega todas as vozes
for item in VOICES_DIR.iterdir():
    if item.is_dir():
        voice_name = item.name
        try:
            voices_registry[voice_name] = load_voice_from_folder(voice_name, item)
        except Exception as e:
            logger.error(f"❌ Voz {voice_name}: {e}")

for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voices_registry:
        continue
    json_file = onnx_file.with_suffix(".onnx.json")
    if json_file.exists():
        try:
            voices_registry[voice_name] = load_voice_from_folder(voice_name, VOICES_DIR)
        except Exception as e:
            logger.error(f"❌ Voz raiz {voice_name}: {e}")

logger.info(f"Total de vozes carregadas: {len(voices_registry)}")
MODEL_LOADED = len(voices_registry) > 0

# =============================================================================
# Mixagem – executada em processo separado
# =============================================================================
def mixing_task(
    ordered_items: list,
    ambient_bytes: Optional[bytes],
    ambient_volume_db: float,
    target_dbfs: float = -20.0
) -> bytes:
    """
    ordered_items: lista de dicts com {'type':'speech','pcm':bytes,'sample_rate':int}
                   ou {'type':'effect','wav_bytes':bytes}
    Retorna bytes do áudio WebM.
    """
    log = logging.getLogger("mixing")
    log.info(f"Iniciando mixagem de {len(ordered_items)} segmentos")
    audio_chunks = []

    for i, item in enumerate(ordered_items):
        if item['type'] == 'speech':
            seg = AudioSegment(
                data=item['pcm'],
                sample_width=2,
                frame_rate=item['sample_rate'],
                channels=1
            )
            if item['sample_rate'] != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
            log.debug(f"Segmento de fala {i}: {len(seg)/1000:.2f}s")

        elif item['type'] == 'effect':
            seg = AudioSegment.from_wav(io.BytesIO(item['wav_bytes']))
            if seg.frame_rate != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
            log.debug(f"Efeito {i}: {len(seg)/1000:.2f}s")

    if not audio_chunks:
        raise ValueError("Nenhum áudio para mixar")

    # Concatenação eficiente
    combined = sum(audio_chunks, AudioSegment.empty())
    log.info(f"Áudio combinado: {len(combined)/1000:.2f}s, dBFS={combined.dBFS:.1f}")

    # Normalização
    if combined.dBFS != target_dbfs:
        gain = target_dbfs - combined.dBFS
        log.info(f"Aplicando ganho de {gain:.1f} dB")
        combined = combined.apply_gain(gain)

    # Ambiente
    if ambient_bytes:
        log.info("Mixando ambiente")
        ambient = AudioSegment.from_wav(io.BytesIO(ambient_bytes))
        if ambient.frame_rate != combined.frame_rate:
            ambient = ambient.set_frame_rate(combined.frame_rate)
        # Ajusta duração
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)

    # Exporta WebM
    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
    log.info(f"Mixagem concluída: {buf.tell()} bytes")
    return buf.getvalue()

# =============================================================================
# Modelos Pydantic
# =============================================================================
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class TTSRequest(BaseModel):
    voice: Optional[str] = Field(None, description="Nome da voz (modo único)")
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="Piper TTS API GPU (pipeline otimizado)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio_geral = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...' | efeitos={list(req.effects.keys())} | ambient={req.ambient.enabled}")

    # ---------- 1. Parse do texto e mapeamento de falantes ----------
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório no modo simples")
        if req.voice not in voices_registry:
            raise HTTPException(404, f"Voz '{req.voice}' não encontrada")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, ns, nw)
        for role, (v, _, _, _) in speaker_map.items():
            if v not in voices_registry:
                raise HTTPException(404, f"Voz '{v}' do speaker '{role}' não encontrada")
        current_role = None

    # ---------- 2. Dividir texto e classificar partes ----------
    parts = re.split(r'(\[.*?\])', req.text)
    logger.info(f"Texto dividido em {len(parts)} partes")

    ordered_items = []          # descreve a sequência final
    synthesis_tasks = []        # (índice_no_ordered, voice_name, texto, params)
    effect_preload = {}         # cache de bytes de efeitos (para evitar I/O repetida nos workers)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Tag de diálogo
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
                logger.info(f"  → Speaker: {role} ({speaker_map[role][0]})")
            continue

        # Efeito sonoro
        if part in req.effects:
            effect_name = req.effects[part]
            logger.info(f"  → Efeito: {part} -> {effect_name}")

            # Carrega o WAV uma vez e guarda bytes
            if effect_name not in effect_preload:
                # Procura na pasta da voz atual (se disponível) ou global
                voice_name_for_effect = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                voice_dir = voices_registry[voice_name_for_effect].model_path.parent if voice_name_for_effect in voices_registry else None
                effect_path = None
                if voice_dir:
                    candidate = Path(voice_dir) / effect_name
                    if candidate.exists():
                        effect_path = candidate
                if not effect_path:
                    candidate = EFFECTS_DIR / effect_name
                    if candidate.exists():
                        effect_path = candidate
                if not effect_path:
                    raise HTTPException(404, f"Efeito '{effect_name}' não encontrado")
                with open(effect_path, "rb") as f:
                    effect_preload[effect_name] = f.read()
                logger.debug(f"Efeito carregado: {len(effect_preload[effect_name])} bytes")
            ordered_items.append({'type': 'effect', 'wav_bytes': effect_preload[effect_name]})
            continue

        # Texto normal → síntese
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto. Use [papel].")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        idx = len(ordered_items)
        ordered_items.append({'type': 'speech', 'pcm': None, 'sample_rate': None})  # placeholder
        synthesis_tasks.append((idx, voice_name, part, speed, noise_s, noise_w))

    if not synthesis_tasks and not any(item['type'] == 'speech' for item in ordered_items):
        raise HTTPException(400, "Nenhum texto para sintetizar")

    # ---------- 3. Disparar sínteses em paralelo (thread pool) ----------
    logger.info(f"Disparando {len(synthesis_tasks)} sínteses em paralelo (executor de {SYNTHESIS_THREADS} threads)")
    futures = {}
    for idx, voice_name, text, speed, noise_s, noise_w in synthesis_tasks:
        pool = voices_registry[voice_name]  # PiperProcessPool
        fut = synthesis_executor.submit(pool.synthesize, text, speed, noise_s, noise_w)
        futures[fut] = idx

    # Coletar resultados
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            pcm, sr = fut.result()
            ordered_items[idx]['pcm'] = pcm
            ordered_items[idx]['sample_rate'] = sr
            logger.debug(f"Síntese {idx} concluída ({len(pcm)} bytes, {sr} Hz)")
        except Exception as e:
            logger.error(f"Síntese {idx} falhou: {e}")
            raise HTTPException(500, f"Erro na síntese de voz: {e}")

    tempo_sintese = time.perf_counter() - inicio_geral
    logger.info(f"Sínteses concluídas em {tempo_sintese:.2f}s")

    # ---------- 4. Preparar ambiente (carregar WAV uma vez) ----------
    ambient_bytes = None
    if req.ambient.enabled and req.ambient.file:
        ambient_path = AMBIENT_DIR / f"{req.ambient.file}.wav"
        if not ambient_path.exists():
            raise HTTPException(404, f"Ambiente '{req.ambient.file}.wav' não encontrado")
        with open(ambient_path, "rb") as f:
            ambient_bytes = f.read()
        logger.info(f"Ambiente carregado: {len(ambient_bytes)} bytes")

    # ---------- 5. Submeter mixagem ao ProcessPoolExecutor ----------
    logger.info(f"Enviando mixagem para pool de {MIXING_PROCESSES} processos")
    loop = asyncio.get_event_loop()
    try:
        webm_bytes = await loop.run_in_executor(
            mixing_executor,
            mixing_task,
            ordered_items,
            ambient_bytes,
            req.ambient.volume_db,
            -20.0
        )
    except Exception as e:
        logger.error(f"Erro na mixagem: {e}")
        raise HTTPException(500, f"Erro na mixagem: {e}")

    tempo_total = time.perf_counter() - inicio_geral
    duracao_estimada = len(webm_bytes) / 8000  # muito grosseiro
    logger.info(f"✅ Requisição finalizada | total={tempo_total:.2f}s | WebM={len(webm_bytes)} bytes | RTF≈{tempo_total/duracao_estimada:.3f}")
    return Response(content=webm_bytes, media_type="audio/webm")

# ---------- Health checks ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if MODEL_LOADED:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    gpu_ok = all(
        entry.is_alive() if hasattr(entry, "is_alive") else True
        for entry in voices_registry.values()
    ) if voices_registry else False
    return {
        "status": "ok",
        "gpu": gpu_ok,
        "voices_loaded": list(voices_registry.keys()),
        "total_voices": len(voices_registry)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
