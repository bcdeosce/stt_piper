#!/usr/bin/env python3
"""
Piper TTS API – GPU via binário compilado
Leitura robusta de JSON + raw audio byte‑a‑byte
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# =============================================================================
# Funções auxiliares de diagnóstico
# =============================================================================
def run_cmd(cmd: list, timeout: int = 10) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (res.stdout or "") + (res.stderr or "")
    except Exception as e:
        return f"Erro: {e}"

def find_file(filename: str, search_paths: list = None) -> Optional[str]:
    if search_paths:
        for p in search_paths:
            full = os.path.join(p, filename)
            if os.path.isfile(full):
                return full
    try:
        res = subprocess.run(["find", "/", "-name", filename, "-type", "f"],
                             capture_output=True, text=True, timeout=20)
        if res.stdout.strip():
            return res.stdout.strip().split("\n")[0]
    except:
        pass
    return None

# =============================================================================
# Configuração de logging
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("piper-api")

# =============================================================================
# DIAGNÓSTICO INICIAL
# =============================================================================
logger.info("=" * 70)
logger.info("DIAGNÓSTICO DE ARQUIVOS E BIBLIOTECAS")
logger.info("=" * 70)

REQUIRED_FILES = {
    "Piper binary": "/app/piper-bin",
    "ONNX Runtime lib": "libonnxruntime.so",
    "Piper Phonemize lib": "libpiper_phonemize.so",
    "eSpeak NG lib": "libespeak-ng.so",
}

files_status = {}
for label, filename in REQUIRED_FILES.items():
    if '/' not in filename and not os.path.isabs(filename):
        found = find_file(filename, os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    elif os.path.isabs(filename):
        found = filename if os.path.isfile(filename) else None
    else:
        found = find_file(filename)
    files_status[label] = found
    if found:
        logger.info(f"✅ {label}: {found}")
    else:
        logger.critical(f"❌ {label}: NÃO ENCONTRADO")

missing_libs = [label for label, path in files_status.items() if not path and 'lib' in label.lower()]
if missing_libs:
    logger.warning("Tentando fallback: busca profunda das libs faltantes...")
    for label in missing_libs:
        libname = REQUIRED_FILES[label]
        found = find_file(libname)
        if found:
            lib_dir = os.path.dirname(found)
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + current_ld
                logger.warning(f"Adicionado ao LD_LIBRARY_PATH: {lib_dir}")
            files_status[label] = found
            logger.info(f"✅ {label} encontrado em fallback: {found}")
        else:
            logger.critical(f"❌ {label} não encontrado mesmo após busca geral.")

voice_model_files = []
for item in Path("/app/voices").rglob("*.onnx"):
    voice_model_files.append(str(item))
for item in Path("/app/voices").rglob("*.onnx.json"):
    voice_model_files.append(str(item))
logger.info(f"Modelos de voz encontrados: {len(voice_model_files)} arquivos")

gpu_info = run_cmd(["nvidia-smi"], timeout=5)
logger.info("GPU info:\n" + gpu_info)

piper_help = run_cmd(["/app/piper-bin", "--help"], timeout=5)
logger.info("piper --help (primeiras linhas):\n" + piper_help[:500])

logger.info("=" * 70)

# =============================================================================
# Diretórios e configurações
# =============================================================================
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

for d in (VOICES_DIR, AMBIENT_DIR, EFFECTS_DIR):
    d.mkdir(exist_ok=True)

SYNTHESIS_THREADS = int(os.getenv("SYNTHESIS_THREADS", "8"))
MIXING_PROCESSES = int(os.getenv("MIXING_PROCESSES", "8"))
PIPER_POOL_SIZE = int(os.getenv("PIPER_POOL_SIZE", "2"))

synthesis_executor = ThreadPoolExecutor(max_workers=SYNTHESIS_THREADS)
mixing_executor = ProcessPoolExecutor(max_workers=MIXING_PROCESSES)

# =============================================================================
# PiperProcess com leitura robusta (byte‑a‑byte)
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

        time.sleep(2.0)
        if self.process.poll() is not None:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Processo morreu na inicialização. stderr: {stderr_tail[-500:]}")
            raise RuntimeError(f"Piper morreu ao iniciar. stderr: {stderr_tail[-200:]}")

        logger.info(f"[PIPER] Pronto (PID {self.process.pid}) para {Path(self.model_path).name}")

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

                # LEITURA ROBUSTA: byte a byte até '\n'
                line_bytes = b""
                while True:
                    ch = self.process.stdout.read(1)
                    if not ch:
                        raise RuntimeError("Processo piper morreu antes de enviar JSON")
                    if ch == b'\n':
                        break
                    line_bytes += ch

                response = json.loads(line_bytes.decode('utf-8'))
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
# Pool de processos Piper
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
# WARM-UP
# =============================================================================
logger.info("=" * 70)
logger.info("🔥 WARM-UP")
warmup_success = False
if "faber" in voices_registry:
    try:
        pool = voices_registry["faber"]
        pcm, sr = pool.synthesize("Teste de aquecimento", 1.0, 0.667, 0.8)
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
        if sr != 22050:
            seg = seg.set_frame_rate(22050)
        buf = io.BytesIO()
        seg.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
        warmup_success = True
        logger.info(f"✅ Warm-up concluído: WebM de {buf.tell()} bytes")
    except Exception as e:
        logger.critical(f"❌ Warm-up falhou: {e}")
else:
    logger.warning("Voz 'faber' não encontrada, warm-up ignorado.")

logger.info("=" * 70)

# =============================================================================
# Mixagem para ProcessPoolExecutor
# =============================================================================
def mixing_task(ordered_items, ambient_bytes, ambient_volume_db, target_dbfs=-20.0):
    import logging as mix_log
    mix_log.basicConfig(level=logging.INFO)
    log = mix_log.getLogger("mixing")
    log.info(f"Mixando {len(ordered_items)} itens")
    audio_chunks = []
    for item in ordered_items:
        if item['type'] == 'speech':
            seg = AudioSegment(data=item['pcm'], sample_width=2, frame_rate=item['sample_rate'], channels=1)
            if item['sample_rate'] != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
        elif item['type'] == 'effect':
            seg = AudioSegment.from_wav(io.BytesIO(item['wav_bytes']))
            if seg.frame_rate != 22050:
                seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
    combined = sum(audio_chunks, AudioSegment.empty())
    if combined.dBFS != target_dbfs:
        combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_bytes:
        ambient = AudioSegment.from_wav(io.BytesIO(ambient_bytes))
        if ambient.frame_rate != combined.frame_rate:
            ambient = ambient.set_frame_rate(combined.frame_rate)
        if len(ambient) < len(combined):
            ambient = ambient * (len(combined) // len(ambient) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)
    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
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
    voice: Optional[str] = Field(None)
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
app = FastAPI(title="Piper TTS API GPU (robusta)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...'")

    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório")
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

    parts = re.split(r'(\[.*?\])', req.text)
    ordered_items = []
    synthesis_tasks = []
    effect_cache = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue
        if part in req.effects:
            effect_name = req.effects[part]
            if effect_name not in effect_cache:
                voice_dir = None
                if is_dialog and current_role:
                    voice_dir = Path(voices_registry[speaker_map[current_role][0]].model_path).parent
                elif req.voice:
                    voice_dir = Path(voices_registry[req.voice].model_path).parent
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
                    effect_cache[effect_name] = f.read()
            ordered_items.append({'type': 'effect', 'wav_bytes': effect_cache[effect_name]})
            continue

        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Speaker não definido")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        idx = len(ordered_items)
        ordered_items.append({'type': 'speech', 'pcm': None, 'sample_rate': None})
        synthesis_tasks.append((idx, voice_name, part, speed, noise_s, noise_w))

    if not synthesis_tasks:
        raise HTTPException(400, "Nenhum texto para sintetizar")

    futures = {}
    for idx, vname, txt, sp, ns, nw in synthesis_tasks:
        pool = voices_registry[vname]
        fut = synthesis_executor.submit(pool.synthesize, txt, sp, ns, nw)
        futures[fut] = idx
    for fut in as_completed(futures):
        idx = futures[fut]
        pcm, sr = fut.result()
        ordered_items[idx]['pcm'] = pcm
        ordered_items[idx]['sample_rate'] = sr

    ambient_bytes = None
    if req.ambient.enabled and req.ambient.file:
        ambient_path = AMBIENT_DIR / f"{req.ambient.file}.wav"
        if not ambient_path.exists():
            raise HTTPException(404, f"Ambiente '{req.ambient.file}.wav' não encontrado")
        with open(ambient_path, "rb") as f:
            ambient_bytes = f.read()

    loop = asyncio.get_event_loop()
    webm = await loop.run_in_executor(
        mixing_executor,
        mixing_task,
        ordered_items,
        ambient_bytes,
        req.ambient.volume_db
    )

    dur = time.perf_counter() - inicio
    logger.info(f"✅ Requisição finalizada em {dur:.2f}s, WebM de {len(webm)} bytes")
    return Response(content=webm, media_type="audio/webm")

# Health checks
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
        "total_voices": len(voices_registry),
        "required_files": files_status,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
