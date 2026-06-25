#!/usr/bin/env python3
"""
Piper TTS API – GPU via binário piper compilado
Leitura robusta: bloco com split pelo primeiro \\n
"""

import os, re, io, json, time, queue, logging, asyncio, subprocess, threading, tempfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("piper-api")

def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except Exception as e:
        return f"Erro: {e}"

# Diagnóstico
logger.info("=" * 70)
logger.info("DIAGNÓSTICO DE AMBIENTE")
logger.info(run_cmd(["nvidia-smi"], timeout=5))
logger.info("=" * 70)

# Diretórios
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
for d in (VOICES_DIR, AMBIENT_DIR, EFFECTS_DIR):
    d.mkdir(exist_ok=True)

SYNTHESIS_THREADS = int(os.getenv("SYNTHESIS_THREADS", "8"))
MIXING_PROCESSES  = int(os.getenv("MIXING_PROCESSES", "8"))
PIPER_POOL_SIZE   = int(os.getenv("PIPER_POOL_SIZE", "2"))

synthesis_executor = ThreadPoolExecutor(max_workers=SYNTHESIS_THREADS)
mixing_executor    = ProcessPoolExecutor(max_workers=MIXING_PROCESSES)
logger.info(f"Paralelismo: {SYNTHESIS_THREADS} threads, {MIXING_PROCESSES} mixers, pool piper={PIPER_POOL_SIZE}")

# =============================================================================
# PiperProcess – robusto, usando split pelo primeiro \n
# =============================================================================
class PiperProcess:
    def __init__(self, model_path, config_path, use_cuda=True):
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
            "--output-raw",
            "--espeak_data", "/app/share/espeak-ng-data"   # <--- CRÍTICO
        ]
        if self.use_cuda:
            cmd.append("--cuda")

        logger.info(f"[PIPER] Iniciando: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        time.sleep(2.0)
        if self.process.poll() is not None:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Morreu ao iniciar. stderr: {stderr_tail[-500:]}")
            raise RuntimeError(f"Piper morreu ao iniciar: {stderr_tail[-200:]}")
        logger.info(f"[PIPER] Pronto (PID {self.process.pid}) para {Path(self.model_path).name}")

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, b''):
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 100:
                self._stderr_lines.pop(0)

    def _get_stderr_tail(self):
        return b"".join(self._stderr_lines).decode(errors='replace')

    def synthesize(self, text, length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8):
        req = json.dumps({"text": text, "length_scale": length_scale,
                          "noise_scale": noise_scale, "noise_w": noise_w_scale}) + "\n"
        with self.lock:
            try:
                self.process.stdin.write(req.encode())
                self.process.stdin.flush()

                # Leitura por bloco até encontrar um '\n'
                chunk = b""
                while b'\n' not in chunk:
                    data = self.process.stdout.read(4096)
                    if not data:
                        raise RuntimeError("Processo piper morreu antes de responder")
                    chunk += data

                # Separa JSON (antes do '\n') e início do áudio (depois)
                json_end = chunk.find(b'\n')
                json_bytes = chunk[:json_end]
                audio_start = chunk[json_end+1:]

                response = json.loads(json_bytes.decode('utf-8'))
                num_samples = response.get("num_samples", 0)
                sample_rate = response.get("sample_rate", 22050)

                # Lê o restante do áudio
                remaining = num_samples * 2 - len(audio_start)
                if remaining > 0:
                    rest_audio = self.process.stdout.read(remaining)
                else:
                    rest_audio = b""
                full_audio = audio_start + rest_audio

                if len(full_audio) != num_samples * 2:
                    logger.warning(f"Tamanho de áudio: esperado {num_samples*2}, obtido {len(full_audio)}")

                return full_audio, sample_rate

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"[PIPER] Erro JSON: {e}. stderr: {self._get_stderr_tail()[-500:]}")
                self._restart()
                raise RuntimeError(f"Falha na comunicação com piper: {e}")
            except Exception as e:
                logger.error(f"[PIPER] Exceção: {e}. stderr: {self._get_stderr_tail()[-500:]}")
                self._restart()
                raise

    def _restart(self):
        logger.warning("[PIPER] Reiniciando...")
        if self.process:
            try: self.process.kill()
            except: pass
            self.process = None
        self._start()

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

# Pool
class PiperProcessPool:
    def __init__(self, model_path, config_path, pool_size=PIPER_POOL_SIZE):
        self.pool = queue.Queue(maxsize=pool_size)
        for i in range(pool_size):
            try:
                proc = PiperProcess(model_path, config_path, use_cuda=True)
                self.pool.put(proc)
                logger.info(f"  → Piper worker {i+1}/{pool_size} para {Path(model_path).stem}")
            except Exception as e:
                logger.error(f"  ✗ Falha ao criar worker {i+1}: {e}")
        if self.pool.empty():
            raise RuntimeError("Nenhum processo Piper iniciado.")

    def synthesize(self, text, length_scale, noise_scale, noise_w_scale):
        proc = self.pool.get()
        try:
            return proc.synthesize(text, length_scale, noise_scale, noise_w_scale)
        finally:
            self.pool.put(proc)

# Carregamento de vozes
voice_pools: Dict[str, PiperProcessPool] = {}

def load_voice(voice_name, voice_dir):
    onnx_files = list(voice_dir.glob("*.onnx"))
    if not onnx_files: raise FileNotFoundError(f"Nenhum .onnx em {voice_dir}")
    model_path = str(onnx_files[0])
    json_path = voice_dir / f"{onnx_files[0].stem}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_dir.glob("*.json"))
        if not json_candidates: raise FileNotFoundError(f"Nenhum .json para {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)
    logger.info(f"Carregando voz {voice_name} (pool de {PIPER_POOL_SIZE})")
    pool = PiperProcessPool(model_path, config_path, pool_size=PIPER_POOL_SIZE)
    voice_pools[voice_name] = pool
    logger.info(f"✅ Voz {voice_name} pronta (GPU)")

for item in VOICES_DIR.iterdir():
    if item.is_dir():
        try: load_voice(item.name, item)
        except Exception as e: logger.error(f"❌ {item.name}: {e}")

for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voice_pools: continue
    json_file = onnx_file.with_suffix(".onnx.json")
    if json_file.exists():
        try: load_voice(voice_name, VOICES_DIR)
        except Exception as e: logger.error(f"❌ {voice_name}: {e}")

logger.info(f"Total de vozes: {len(voice_pools)}")
MODEL_LOADED = len(voice_pools) > 0

# Warm-up
logger.info("=" * 70)
logger.info("🔥 WARM-UP")
warmup_success = False
if "faber" in voice_pools:
    try:
        pcm, sr = voice_pools["faber"].synthesize("Teste de aquecimento", 1.0, 0.667, 0.8)
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
        if sr != 22050: seg = seg.set_frame_rate(22050)
        buf = io.BytesIO()
        seg.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
        warmup_success = True
        logger.info(f"✅ Warm-up OK – WebM de {buf.tell()} bytes")
    except Exception as e:
        logger.critical(f"❌ Warm-up falhou: {e}")
else:
    logger.warning("Voz 'faber' não encontrada, warm-up ignorado.")
logger.info("=" * 70)

# Mixagem (igual)
def mixing_task(ordered_items, ambient_bytes, ambient_volume_db, target_dbfs=-20.0):
    import logging as mix_log
    mix_log.basicConfig(level=logging.INFO)
    log = mix_log.getLogger("mixing")
    log.info(f"Mixando {len(ordered_items)} itens")
    audio_chunks = []
    for item in ordered_items:
        if item['type'] == 'speech':
            seg = AudioSegment(data=item['pcm'], sample_width=2, frame_rate=item['sample_rate'], channels=1)
            if item['sample_rate'] != 22050: seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
        elif item['type'] == 'effect':
            seg = AudioSegment.from_wav(io.BytesIO(item['wav_bytes']))
            if seg.frame_rate != 22050: seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
    if not audio_chunks: raise ValueError("Nenhum áudio")
    combined = sum(audio_chunks, AudioSegment.empty())
    if combined.dBFS != target_dbfs: combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_bytes:
        ambient = AudioSegment.from_wav(io.BytesIO(ambient_bytes))
        if ambient.frame_rate != combined.frame_rate: ambient = ambient.set_frame_rate(combined.frame_rate)
        if len(ambient) < len(combined): ambient = ambient * (len(combined) // len(ambient) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)
    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
    log.info(f"Mixagem concluída: {buf.tell()} bytes")
    return buf.getvalue()

# Modelos Pydantic (mesmos)
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
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# FastAPI
app = FastAPI(title="Piper TTS GPU (robusto)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...'")

    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice: raise HTTPException(400, "Campo 'voice' obrigatório")
        if req.voice not in voice_pools: raise HTTPException(404, f"Voz '{req.voice}' não encontrada")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, ns, nw)
        for role, (v, _, _, _) in speaker_map.items():
            if v not in voice_pools: raise HTTPException(404, f"Voz '{v}' do speaker '{role}' não encontrada")
        current_role = None

    parts = re.split(r'(\[.*?\])', req.text)
    ordered_items = []
    synthesis_tasks = []
    effect_cache = {}

    for part in parts:
        part = part.strip()
        if not part: continue
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map: current_role = role
            continue
        if part in req.effects:
            effect_name = req.effects[part]
            if effect_name not in effect_cache:
                voice_name_eff = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                voice_dir = Path(voice_pools[voice_name_eff].pool.queue[0].model_path).parent
                effect_path = None
                if (voice_dir / effect_name).exists(): effect_path = voice_dir / effect_name
                else:
                    candidate = EFFECTS_DIR / effect_name
                    if candidate.exists(): effect_path = candidate
                if not effect_path: raise HTTPException(404, f"Efeito '{effect_name}' não encontrado")
                with open(effect_path, "rb") as f: effect_cache[effect_name] = f.read()
            ordered_items.append({'type': 'effect', 'wav_bytes': effect_cache[effect_name]})
            continue

        if is_dialog:
            if current_role is None: raise HTTPException(400, "Speaker não definido")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        idx = len(ordered_items)
        ordered_items.append({'type': 'speech', 'pcm': None, 'sample_rate': None})
        synthesis_tasks.append((idx, voice_name, part, speed, noise_s, noise_w))

    if not synthesis_tasks: raise HTTPException(400, "Nenhum texto para sintetizar")

    futures = {}
    for idx, vname, txt, sp, ns, nw in synthesis_tasks:
        pool = voice_pools[vname]
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
        if not ambient_path.exists(): raise HTTPException(404, f"Ambiente '{req.ambient.file}' não encontrado")
        with open(ambient_path, "rb") as f: ambient_bytes = f.read()

    loop = asyncio.get_event_loop()
    webm = await loop.run_in_executor(mixing_executor, mixing_task,
                                      ordered_items, ambient_bytes, req.ambient.volume_db)

    dur = time.perf_counter() - inicio
    logger.info(f"✅ Requisição finalizada em {dur:.2f}s, WebM de {len(webm)} bytes")
    return Response(content=webm, media_type="audio/webm")

# Health checks
@app.get("/started")
async def started(): return Response(status_code=200, content="started")
@app.get("/ready")
async def ready():
    if MODEL_LOADED: return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")
@app.get("/live")
async def live(): return Response(status_code=200, content="alive")
@app.get("/health")
async def health():
    return {"status": "ok", "gpu": True, "warmup": warmup_success,
            "voices_loaded": list(voice_pools.keys()), "total_voices": len(voice_pools)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
