import os
import re
import io
import time
import queue
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment
from piper import PiperVoice, SynthesisConfig

# ---------- Configuração ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-api")

SAMPLE_RATE_TARGET = 22050
MAX_GPU_JOBS = 3
GPU_WORKERS = 2
MIX_WORKERS = 10

# Diretórios
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# Caches
EFFECTS_CACHE: Dict[str, AudioSegment] = {}
AMBIENT_CACHE: Dict[str, AudioSegment] = {}

# Vozes disponíveis (pool de instâncias)
voices_registry: Dict[str, dict] = {}

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
        import json
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
                logger.error(f"❌ Erro ao carregar voz {name}: {e}")
    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name in voices_registry:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                pool = VoicePool(str(onnx_file), str(json_file), pool_size=2)
                voices_registry[name] = {"model_path": str(onnx_file), "config_path": str(json_file), "genero": "Personalizada", "pool": pool, "path": VOICES_DIR}
                logger.info(f"✅ Voz raiz carregada: {name}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar voz {name}: {e}")
    logger.info(f"Total de vozes: {len(voices_registry)}")

# ---------- Efeitos e ambientes ----------
def preload_all_effects():
    if not EFFECTS_DIR.exists(): return
    for wav_file in EFFECTS_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            EFFECTS_CACHE[wav_file.name] = seg
        except Exception as e: logger.error(f"Efeito {wav_file.name}: {e}")

def preload_all_ambient():
    if not AMBIENT_DIR.exists(): return
    for wav_file in AMBIENT_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            AMBIENT_CACHE[wav_file.stem] = seg
        except Exception as e: logger.error(f"Ambiente {wav_file.name}: {e}")

def get_effect(voice_name: str, effect_file: str) -> AudioSegment:
    voice_entry = voices_registry.get(voice_name)
    if voice_entry:
        effect_path = voice_entry["path"] / effect_file
        if effect_path.exists():
            seg = AudioSegment.from_wav(str(effect_path))
            if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            return seg
    if effect_file in EFFECTS_CACHE: return EFFECTS_CACHE[effect_file]
    global_path = EFFECTS_DIR / effect_file
    if global_path.exists():
        seg = AudioSegment.from_wav(str(global_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        EFFECTS_CACHE[effect_file] = seg
        return seg
    raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado")

def get_ambient(ambient_file: str, volume_db: float) -> AudioSegment:
    if ambient_file not in AMBIENT_CACHE:
        ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
        if not ambient_path.exists(): raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")
        seg = AudioSegment.from_wav(str(ambient_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        AMBIENT_CACHE[ambient_file] = seg
    return AMBIENT_CACHE[ambient_file] + volume_db

# ---------- Síntese e mixagem ----------
def synthesize_speech(voice, text: str, speed: float, noise_s: float, noise_w: float) -> AudioSegment:
    config = SynthesisConfig(length_scale=speed, noise_scale=noise_s, noise_w_scale=noise_w)
    chunks = voice.synthesize(text, syn_config=config)
    audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunks)
    sample_rate = voice.config.sample_rate
    seg = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=sample_rate, channels=1)
    if seg.frame_rate != SAMPLE_RATE_TARGET: seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
    return seg

def mix_and_export(segments: List[AudioSegment], ambient_cfg) -> bytes:
    if not segments: raise ValueError("Nenhum segmento")
    combined = AudioSegment.empty()
    for seg in segments: combined += seg
    target_dbfs = -20.0
    if combined.dBFS != target_dbfs: combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_cfg.enabled and ambient_cfg.file:
        ambient = get_ambient(ambient_cfg.file, ambient_cfg.volume_db)
        if len(ambient) < len(combined): ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)
    with io.BytesIO() as buf:
        combined.export(buf, format="wav")
        return buf.getvalue()

# ---------- Modelos de requisição ----------
from pydantic import BaseModel, Field

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
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

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

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    try:
        inicio = time.perf_counter()
        is_dialog = bool(req.speakers)
        if not is_dialog:
            if not req.voice: raise HTTPException(400, "voice obrigatório")
            if req.voice not in voices_registry: raise HTTPException(404, "Voz não encontrada")
            speaker_params = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        else:
            speaker_params = {}
            for spk in req.speakers:
                ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
                nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
                speaker_params[spk.role] = (spk.voice, spk.speed, ns, nw)
            for vn, _, _, _ in speaker_params.values():
                if vn not in voices_registry: raise HTTPException(404, f"Voz '{vn}' não encontrada")

        parts = re.split(r'(\[.*?\])', req.text)
        current_role = None
        segments = []

        for part in parts:
            part = part.strip()
            if not part: continue
            if is_dialog and part.startswith('[') and part.endswith(']'):
                role = part[1:-1]
                if role in speaker_params: current_role = role
                continue
            if part in req.effects:
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                segments.append({"type": "effect", "effect_file": req.effects[part], "voice_name": vname})
                continue
            if is_dialog:
                if current_role is None: raise HTTPException(400, "Speaker não definido")
                voice_name, speed, noise_s, noise_w = speaker_params[current_role]
            else:
                voice_name, speed, noise_s, noise_w = speaker_params[None]
            segments.append({"type": "speech", "voice_name": voice_name, "text": part,
                             "speed": speed, "noise_s": noise_s, "noise_w": noise_w})

        if not segments: raise HTTPException(400, "Nenhum segmento")

        loop = asyncio.get_running_loop()

        async def process_segment(index: int, seg: dict):
            if seg["type"] == "effect":
                return index, get_effect(seg["voice_name"], seg["effect_file"])
            pool = voices_registry[seg["voice_name"]]["pool"]
            async with gpu_semaphore:
                voice = await loop.run_in_executor(gpu_executor, pool.get)
                try:
                    audio = await loop.run_in_executor(gpu_executor, synthesize_speech, voice, seg["text"], seg["speed"], seg["noise_s"], seg["noise_w"])
                finally:
                    pool.put(voice)
            return index, audio

        tasks = [process_segment(i, s) for i, s in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        audio_segments = [seg for _, seg in results]

        final_wav = await loop.run_in_executor(mix_executor, mix_and_export, audio_segments, req.ambient)
        duracao = len(final_wav) / (2 * SAMPLE_RATE_TARGET)
        tempo_total = time.perf_counter() - inicio
        logger.info(f"✅ Síntese finalizada | tempo={tempo_total:.3f}s | áudio={duracao:.2f}s | RTF={tempo_total/duracao:.3f}")
        return Response(content=final_wav, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erro na síntese")
        raise HTTPException(500, "Erro interno")

@app.get("/started")
async def started(): return Response(status_code=200)
@app.get("/ready")
async def ready(): return Response(status_code=200 if voices_registry else 503)
@app.get("/live")
async def live(): return Response(status_code=200)
@app.get("/health")
async def health():
    return {"status": "ok", "gpu": True, "voices": list(voices_registry.keys()), "total": len(voices_registry)}
