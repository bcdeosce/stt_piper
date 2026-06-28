import os
import re
import io
import time
import queue
import asyncio
import logging
import json
import threading
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydub import AudioSegment
from piper import PiperVoice, SynthesisConfig
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# ---------- Logging com buffer para endpoint /logs ----------
class MemoryHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.capacity = capacity
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

memory_handler = MemoryHandler(capacity=100)
memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout), memory_handler])
logger = logging.getLogger("piper-api")
logger.info("Iniciando Piper API")

# ---------- Constantes ----------
SAMPLE_RATE_TARGET = 22050
MAX_GPU_JOBS = 3
GPU_WORKERS = 2
MIX_WORKERS = 10

BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

EFFECTS_CACHE: Dict[str, AudioSegment] = {}
AMBIENT_CACHE: Dict[str, AudioSegment] = {}
voices_registry: Dict[str, dict] = {}

# ---------- Benchmarks ----------
tts_times: list[float] = []
mix_times: list[float] = []
total_times: list[float] = []
bench_lock = threading.Lock()

def record_bench(total: float, tts_list: list[float], mix: float):
    with bench_lock:
        total_times.append(total)
        tts_times.extend(tts_list)
        mix_times.append(mix)

def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "p95": 0, "count": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "count": len(values)
    }

# ---------- Pool de vozes ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=True)
            self.pool.put(voice)

    def get(self, timeout=2.0):
        return self.pool.get(timeout=timeout)

    def put(self, voice):
        self.pool.put(voice)

# ---------- Registo de vozes ----------
def load_voice_from_folder(voice_name: str, voice_path: Path) -> dict:
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_path}")
    model_path = str(onnx_files[0])
    base_name = onnx_files[0].stem
    json_path = voice_path / f"{base_name}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum .json para {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    genero = "Desconhecido"
    meta_path = voice_path / f"{voice_name}.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                genero = meta.get("genero", "Desconhecido")
        except:
            pass

    pool = VoicePool(model_path, config_path, pool_size=2)
    return {"model_path": model_path, "config_path": config_path, "genero": genero, "pool": pool, "path": voice_path}

def load_all_voices():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            name = item.name
            try:
                voices_registry[name] = load_voice_from_folder(name, item)
                logger.info(f"✅ Voz carregada: {name}")
            except Exception as e:
                logger.error(f"❌ Falha ao carregar voz {name}: {e}")

    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name in voices_registry:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                pool = VoicePool(str(onnx_file), str(json_file), pool_size=2)
                voices_registry[name] = {
                    "model_path": str(onnx_file),
                    "config_path": str(json_file),
                    "genero": "Personalizada",
                    "pool": pool,
                    "path": VOICES_DIR
                }
                logger.info(f"✅ Voz raiz carregada: {name}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar voz {name}: {e}")

    logger.info(f"Total de vozes: {len(voices_registry)}")

# ---------- Efeitos e ambientes ----------
def preload_all_effects():
    if not EFFECTS_DIR.exists():
        logger.warning(f"Diretório de efeitos não encontrado: {EFFECTS_DIR}")
        return
    for wav_file in EFFECTS_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            EFFECTS_CACHE[wav_file.name] = seg
            logger.info(f"✔ Efeito pré-carregado: {wav_file.name}")
        except Exception as e:
            logger.error(f"Erro ao carregar efeito {wav_file.name}: {e}")

def preload_all_ambient():
    if not AMBIENT_DIR.exists():
        logger.warning(f"Diretório de ambientes não encontrado: {AMBIENT_DIR}")
        return
    for wav_file in AMBIENT_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            AMBIENT_CACHE[wav_file.stem] = seg
            logger.info(f"✔ Ambiente pré-carregado: {wav_file.stem}")
        except Exception as e:
            logger.error(f"Erro ao carregar ambiente {wav_file.name}: {e}")

def get_effect(voice_name: str, effect_file: str) -> AudioSegment:
    logger.info(f"Procurando efeito '{effect_file}' para voz '{voice_name}'")
    voice_entry = voices_registry.get(voice_name)
    if voice_entry:
        effect_path = voice_entry["path"] / effect_file
        logger.info(f"Tentando caminho local: {effect_path}")
        if effect_path.exists():
            seg = AudioSegment.from_wav(str(effect_path))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            return seg

    if effect_file in EFFECTS_CACHE:
        logger.info("Efeito encontrado no cache global")
        return EFFECTS_CACHE[effect_file]

    global_path = EFFECTS_DIR / effect_file
    logger.info(f"Tentando caminho global: {global_path}")
    if global_path.exists():
        seg = AudioSegment.from_wav(str(global_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        EFFECTS_CACHE[effect_file] = seg
        return seg

    raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado")

def get_ambient(ambient_file: str, volume_db: float) -> AudioSegment:
    logger.info(f"Procurando ambiente '{ambient_file}'")
    if ambient_file not in AMBIENT_CACHE:
        ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
        logger.info(f"Tentando caminho: {ambient_path}")
        if not ambient_path.exists():
            raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado em {AMBIENT_DIR}")
        seg = AudioSegment.from_wav(str(ambient_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        AMBIENT_CACHE[ambient_file] = seg
    return AMBIENT_CACHE[ambient_file] + volume_db

# ---------- Síntese ----------
def synthesize_speech(voice, text: str, speed: float,
                      noise_s: float, noise_w: float) -> Tuple[AudioSegment, float]:
    t_start = time.perf_counter()
    config = SynthesisConfig(length_scale=speed, noise_scale=noise_s, noise_w_scale=noise_w)
    chunks = voice.synthesize(text, syn_config=config)
    audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunks)
    sample_rate = voice.config.sample_rate
    t_end = time.perf_counter()
    seg = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=sample_rate, channels=1)
    if seg.frame_rate != SAMPLE_RATE_TARGET:
        seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
    return seg, t_end - t_start

def mix_and_export(segments: List[AudioSegment], ambient_cfg) -> Tuple[bytes, float]:
    t_start = time.perf_counter()
    if not segments:
        raise ValueError("Nenhum segmento")
    combined = AudioSegment.empty()
    for seg in segments:
        combined += seg
    target_dbfs = -20.0
    if combined.dBFS != target_dbfs:
        combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_cfg.enabled and ambient_cfg.file:
        ambient = get_ambient(ambient_cfg.file, ambient_cfg.volume_db)
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)
    with io.BytesIO() as buf:
        combined.export(buf, format="wav")
        wav_bytes = buf.getvalue()
    t_end = time.perf_counter()
    return wav_bytes, t_end - t_start

# ---------- Modelos ----------
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

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS API (GPU 1.4.2)")

gpu_executor: Optional[ThreadPoolExecutor] = None
mix_executor: Optional[ThreadPoolExecutor] = None
gpu_semaphore: asyncio.Semaphore = None

@app.on_event("startup")
async def startup():
    global gpu_executor, mix_executor, gpu_semaphore
    logger.info("Pré-carregando recursos...")
    preload_all_effects()
    preload_all_ambient()
    load_all_voices()
    gpu_executor = ThreadPoolExecutor(max_workers=GPU_WORKERS)
    mix_executor = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    gpu_semaphore = asyncio.Semaphore(MAX_GPU_JOBS)
    logger.info("Sistema pronto (piper-tts 1.4.2 + GPU).")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    try:
        t_total_start = time.perf_counter()
        logger.info(f"Síntese solicitada: text='{req.text[:50]}...'")
        is_dialog = bool(req.speakers)

        if not is_dialog:
            if not req.voice:
                raise HTTPException(400, "Campo 'voice' é obrigatório")
            if req.voice not in voices_registry:
                raise HTTPException(404, f"Voz '{req.voice}' não encontrada")
            speaker_params = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        else:
            speaker_params = {}
            for spk in req.speakers:
                ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
                nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
                speaker_params[spk.role] = (spk.voice, spk.speed, ns, nw)
            for vn, _, _, _ in speaker_params.values():
                if vn not in voices_registry:
                    raise HTTPException(404, f"Voz '{vn}' não encontrada")

        parts = re.split(r'(\[.*?\])', req.text)
        current_role = None
        segments = []

        for part in parts:
            part = part.strip()
            if not part: continue
            if is_dialog and part.startswith('[') and part.endswith(']'):
                role = part[1:-1]
                if role in speaker_params:
                    current_role = role
                continue
            if part in req.effects:
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                segments.append({"type": "effect", "effect_file": req.effects[part], "voice_name": vname})
                continue
            if is_dialog:
                if current_role is None:
                    raise HTTPException(400, "Nenhum speaker definido")
                voice_name, speed, noise_s, noise_w = speaker_params[current_role]
            else:
                voice_name, speed, noise_s, noise_w = speaker_params[None]
            segments.append({
                "type": "speech", "voice_name": voice_name, "text": part,
                "speed": speed, "noise_s": noise_s, "noise_w": noise_w
            })

        if not segments:
            raise HTTPException(400, "Nenhum segmento")

        loop = asyncio.get_running_loop()

        async def process_segment(index: int, seg: dict) -> Tuple[int, AudioSegment, float]:
            if seg["type"] == "effect":
                try:
                    eff = get_effect(seg["voice_name"], seg["effect_file"])
                    return index, eff, 0.0
                except Exception as e:
                    logger.error(f"Erro ao carregar efeito {seg['effect_file']}: {e}")
                    return index, AudioSegment.silent(duration=0), 0.0
            pool = voices_registry[seg["voice_name"]]["pool"]
            async with gpu_semaphore:
                voice = await loop.run_in_executor(gpu_executor, pool.get)
                try:
                    audio, tts_time = await loop.run_in_executor(
                        gpu_executor, synthesize_speech, voice,
                        seg["text"], seg["speed"], seg["noise_s"], seg["noise_w"]
                    )
                finally:
                    pool.put(voice)
            return index, audio, tts_time

        tasks = [process_segment(i, s) for i, s in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        audio_segments = [seg for _, seg, _ in results if len(seg) > 0]
        segment_tts_times = [t for _, _, t in results]

        wav_bytes, mix_time = await loop.run_in_executor(
            mix_executor, mix_and_export, audio_segments, req.ambient
        )

        t_total_end = time.perf_counter()
        total_time = t_total_end - t_total_start
        record_bench(total_time, segment_tts_times, mix_time)

        duracao = len(wav_bytes) / (2 * SAMPLE_RATE_TARGET)
        logger.info(f"✅ Síntese concluída | total={total_time:.3f}s | tts={sum(segment_tts_times):.3f}s | mix={mix_time:.3f}s | áudio={duracao:.2f}s")
        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Erro na síntese")
        raise HTTPException(500, "Erro interno")

# ---------- Endpoints de saúde ----------
@app.get("/started")
async def started(): return Response(status_code=200, content="started")
@app.get("/ready")
async def ready(): return Response(status_code=200 if voices_registry else 503, content="ready" if voices_registry else "loading")
@app.get("/live")
async def live(): return Response(status_code=200, content="alive")
@app.get("/health")
async def health():
    return {"status": "ok", "gpu": True, "voices": list(voices_registry.keys()), "total": len(voices_registry)}

# ---------- Benchmark ----------
@app.get("/bench")
async def bench():
    with bench_lock:
        tts_stats = compute_stats(tts_times)
        mix_stats = compute_stats(mix_times)
        total_stats = compute_stats(total_times)
    return {
        "benchmark_results": {
            "total": total_stats,
            "tts": tts_stats,
            "mix": mix_stats,
            "total_tts": tts_stats
        },
        "configuration": {
            "status": "ok",
            "voices": list(voices_registry.keys()),
            "workers": {"tts": GPU_WORKERS, "mix": MIX_WORKERS},
            "gpu": True,
            "precision": "fp32",
            "num_step": 0,
            "batch_timeout": 0.0,
            "max_batch_size": 1
        }
    }

# ---------- Logs ----------
@app.get("/logs")
async def get_logs():
    return {"logs": memory_handler.buffer}

# ---------- Diagnóstico GPU ----------
@app.get("/gpu")
async def gpu_diagnostics():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    nvidia_smi = ""
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        nvidia_smi = str(e)
    return {
        "onnxruntime_version": ort.__version__,
        "providers": providers,
        "device": ort.get_device(),
        "nvidia_smi": nvidia_smi.strip(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "voices_loaded": list(voices_registry.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)import os
import re
import io
import time
import queue
import asyncio
import logging
import json
import threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydub import AudioSegment
from piper import PiperVoice, SynthesisConfig
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------- Configuração ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-api")

SAMPLE_RATE_TARGET = 22050
MAX_GPU_JOBS = 3          # sínteses simultâneas na GPU
GPU_WORKERS = 2            # threads do pool da GPU
MIX_WORKERS = 10           # threads do pool de mixagem

# Diretórios
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# Caches de efeitos e ambientes
EFFECTS_CACHE: Dict[str, AudioSegment] = {}
AMBIENT_CACHE: Dict[str, AudioSegment] = {}

# Vozes carregadas
voices_registry: Dict[str, dict] = {}

# ---------- Benchmarks ----------
tts_times: list[float] = []      # tempos de cada síntese individual
mix_times: list[float] = []      # tempos de cada mixagem
total_times: list[float] = []    # tempos totais de cada requisição
bench_lock = threading.Lock()

def record_bench(total: float, tts_list: list[float], mix: float):
    """Regista os tempos de uma requisição."""
    with bench_lock:
        total_times.append(total)
        tts_times.extend(tts_list)
        mix_times.append(mix)

def compute_stats(values: list[float]) -> dict:
    """Calcula mean, min, max, p95 e count."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "p95": 0, "count": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "count": len(values)
    }

# ---------- Pool de vozes (GPU) ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=True)
            self.pool.put(voice)

    def get(self, timeout=2.0):
        return self.pool.get(timeout=timeout)

    def put(self, voice):
        self.pool.put(voice)

# ---------- Registo de vozes ----------
def load_voice_from_folder(voice_name: str, voice_path: Path) -> dict:
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_path}")
    model_path = str(onnx_files[0])
    base_name = onnx_files[0].stem
    json_path = voice_path / f"{base_name}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum .json para {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    genero = "Desconhecido"
    meta_path = voice_path / f"{voice_name}.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                genero = meta.get("genero", "Desconhecido")
        except:
            pass

    pool = VoicePool(model_path, config_path, pool_size=2)
    return {
        "model_path": model_path,
        "config_path": config_path,
        "genero": genero,
        "pool": pool,
        "path": voice_path
    }

def load_all_voices():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            name = item.name
            try:
                voices_registry[name] = load_voice_from_folder(name, item)
                logger.info(f"✅ Voz carregada: {name}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar voz {name}: {e}")

    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name in voices_registry:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                pool = VoicePool(str(onnx_file), str(json_file), pool_size=2)
                voices_registry[name] = {
                    "model_path": str(onnx_file),
                    "config_path": str(json_file),
                    "genero": "Personalizada",
                    "pool": pool,
                    "path": VOICES_DIR
                }
                logger.info(f"✅ Voz raiz carregada: {name}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar voz {name}: {e}")

    logger.info(f"Total de vozes: {len(voices_registry)}")

# ---------- Efeitos e ambientes ----------
def preload_all_effects():
    if not EFFECTS_DIR.exists():
        return
    for wav_file in EFFECTS_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            EFFECTS_CACHE[wav_file.name] = seg
        except Exception as e:
            logger.error(f"Efeito {wav_file.name}: {e}")

def preload_all_ambient():
    if not AMBIENT_DIR.exists():
        return
    for wav_file in AMBIENT_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            AMBIENT_CACHE[wav_file.stem] = seg
        except Exception as e:
            logger.error(f"Ambiente {wav_file.name}: {e}")

def get_effect(voice_name: str, effect_file: str) -> AudioSegment:
    voice_entry = voices_registry.get(voice_name)
    if voice_entry:
        effect_path = voice_entry["path"] / effect_file
        if effect_path.exists():
            seg = AudioSegment.from_wav(str(effect_path))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            return seg
    if effect_file in EFFECTS_CACHE:
        return EFFECTS_CACHE[effect_file]
    global_path = EFFECTS_DIR / effect_file
    if global_path.exists():
        seg = AudioSegment.from_wav(str(global_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        EFFECTS_CACHE[effect_file] = seg
        return seg
    raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado")

def get_ambient(ambient_file: str, volume_db: float) -> AudioSegment:
    if ambient_file not in AMBIENT_CACHE:
        ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
        if not ambient_path.exists():
            raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")
        seg = AudioSegment.from_wav(str(ambient_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        AMBIENT_CACHE[ambient_file] = seg
    return AMBIENT_CACHE[ambient_file] + volume_db

# ---------- Síntese e mixagem ----------
def synthesize_speech(voice, text: str, speed: float,
                      noise_s: float, noise_w: float) -> Tuple[AudioSegment, float]:
    """Retorna o áudio e o tempo gasto na síntese."""
    t_start = time.perf_counter()
    config = SynthesisConfig(length_scale=speed, noise_scale=noise_s, noise_w_scale=noise_w)
    chunks = voice.synthesize(text, syn_config=config)
    audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunks)
    sample_rate = voice.config.sample_rate
    t_end = time.perf_counter()
    seg = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=sample_rate, channels=1)
    if seg.frame_rate != SAMPLE_RATE_TARGET:
        seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
    return seg, t_end - t_start

def mix_and_export(segments: List[AudioSegment], ambient_cfg) -> Tuple[bytes, float]:
    """Retorna o áudio final e o tempo gasto na mixagem."""
    t_start = time.perf_counter()
    if not segments:
        raise ValueError("Nenhum segmento")
    combined = AudioSegment.empty()
    for seg in segments:
        combined += seg
    target_dbfs = -20.0
    if combined.dBFS != target_dbfs:
        combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_cfg.enabled and ambient_cfg.file:
        ambient = get_ambient(ambient_cfg.file, ambient_cfg.volume_db)
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)
    with io.BytesIO() as buf:
        combined.export(buf, format="wav")
        wav_bytes = buf.getvalue()
    t_end = time.perf_counter()
    return wav_bytes, t_end - t_start

# ---------- Modelos ----------
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

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS API (GPU 1.4.2)")

gpu_executor: Optional[ThreadPoolExecutor] = None
mix_executor: Optional[ThreadPoolExecutor] = None
gpu_semaphore: asyncio.Semaphore = None

@app.on_event("startup")
async def startup():
    global gpu_executor, mix_executor, gpu_semaphore
    preload_all_effects()
    preload_all_ambient()
    load_all_voices()
    gpu_executor = ThreadPoolExecutor(max_workers=GPU_WORKERS)
    mix_executor = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    gpu_semaphore = asyncio.Semaphore(MAX_GPU_JOBS)
    logger.info("Sistema pronto (piper-tts 1.4.2 + GPU).")

# ---------- Endpoint principal ----------
@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    try:
        t_total_start = time.perf_counter()
        is_dialog = bool(req.speakers)

        # Mapeamento de speakers
        if not is_dialog:
            if not req.voice:
                raise HTTPException(400, "Campo 'voice' é obrigatório no modo simples")
            if req.voice not in voices_registry:
                raise HTTPException(404, f"Voz não encontrada: {req.voice}")
            speaker_params = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        else:
            speaker_params = {}
            for spk in req.speakers:
                ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
                nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
                speaker_params[spk.role] = (spk.voice, spk.speed, ns, nw)
            for vn, _, _, _ in speaker_params.values():
                if vn not in voices_registry:
                    raise HTTPException(404, f"Voz '{vn}' não encontrada")

        # Divisão do texto
        parts = re.split(r'(\[.*?\])', req.text)
        current_role = None
        segments = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if is_dialog and part.startswith('[') and part.endswith(']'):
                role = part[1:-1]
                if role in speaker_params:
                    current_role = role
                continue
            if part in req.effects:
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                segments.append({"type": "effect", "effect_file": req.effects[part], "voice_name": vname})
                continue
            if is_dialog:
                if current_role is None:
                    raise HTTPException(400, "Nenhum speaker definido antes do texto.")
                voice_name, speed, noise_s, noise_w = speaker_params[current_role]
            else:
                voice_name, speed, noise_s, noise_w = speaker_params[None]
            segments.append({
                "type": "speech",
                "voice_name": voice_name,
                "text": part,
                "speed": speed,
                "noise_s": noise_s,
                "noise_w": noise_w
            })

        if not segments:
            raise HTTPException(400, "Nenhum segmento de áudio gerado.")

        loop = asyncio.get_running_loop()

        # Processamento paralelo dos segmentos
        async def process_segment(index: int, seg: dict) -> Tuple[int, AudioSegment, float]:
            if seg["type"] == "effect":
                return index, get_effect(seg["voice_name"], seg["effect_file"]), 0.0
            pool = voices_registry[seg["voice_name"]]["pool"]
            async with gpu_semaphore:
                voice = await loop.run_in_executor(gpu_executor, pool.get)
                try:
                    audio, tts_time = await loop.run_in_executor(
                        gpu_executor, synthesize_speech, voice,
                        seg["text"], seg["speed"], seg["noise_s"], seg["noise_w"]
                    )
                finally:
                    pool.put(voice)
            return index, audio, tts_time

        tasks = [process_segment(i, s) for i, s in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        audio_segments = [seg for _, seg, _ in results]
        segment_tts_times = [t for _, _, t in results]

        # Mixagem final
        wav_bytes, mix_time = await loop.run_in_executor(
            mix_executor, mix_and_export, audio_segments, req.ambient
        )

        t_total_end = time.perf_counter()
        total_time = t_total_end - t_total_start

        # Registar benchmark
        record_bench(total_time, segment_tts_times, mix_time)

        duracao = len(wav_bytes) / (2 * SAMPLE_RATE_TARGET)
        logger.info(
            f"✅ Síntese finalizada | total={total_time:.3f}s | "
            f"tts={sum(segment_tts_times):.3f}s | mix={mix_time:.3f}s | "
            f"áudio={duracao:.2f}s | RTF={total_time/duracao:.3f}"
        )
        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Erro na síntese")
        raise HTTPException(500, "Erro interno")

# ---------- Endpoints de saúde ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if voices_registry:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": True,
        "voices": list(voices_registry.keys()),
        "total": len(voices_registry)
    }

# ---------- Benchmark endpoint ----------
@app.get("/bench")
async def bench():
    with bench_lock:
        tts_stats = compute_stats(tts_times)
        mix_stats = compute_stats(mix_times)
        total_stats = compute_stats(total_times)
    return {
        "benchmark_results": {
            "total": total_stats,
            "tts": tts_stats,
            "mix": mix_stats,
            "total_tts": tts_stats  # mesmo que tts, pois não temos métrica separada de "total_tts"
        },
        "configuration": {
            "status": "ok",
            "voices": list(voices_registry.keys()),
            "workers": {
                "tts": GPU_WORKERS,
                "mix": MIX_WORKERS
            },
            "gpu": True,
            "precision": "fp32",      # ONNX Runtime usa float32 por padrão
            "num_step": 0,            # não aplicável ao Piper
            "batch_timeout": 0.0,
            "max_batch_size": 1
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
