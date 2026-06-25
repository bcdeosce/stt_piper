import os
import re
import io
import wave
import time
import queue
import asyncio
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from pydub import AudioSegment

# ---------- Configurações ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-api")

# Constantes
MAX_GPU_JOBS = 1                # Apenas uma síntese de cada vez (binário usa GPU intensivamente)
GPU_WORKERS = 1                 # 1 processo dedicado à GPU
MIX_WORKERS = 10                # threads para mixagem
SAMPLE_RATE_TARGET = 22050
PIPER_BIN = "/app/piper/piper"  # binário compilado com CUDA

# Diretórios
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# Cache de efeitos e ambientes
EFFECTS_CACHE: Dict[str, AudioSegment] = {}
AMBIENT_CACHE: Dict[str, AudioSegment] = {}

# Pool de processos para GPU (executa o binário piper)
gpu_pool: Optional[ProcessPoolExecutor] = None
# Pool de threads para mixagem
mix_pool: Optional[ThreadPoolExecutor] = None
# Semáforo para controlar acesso concorrente à GPU
gpu_semaphore: asyncio.Semaphore = None

# Vozes disponíveis (apenas metadados, sem carregar modelos)
voices_registry: Dict[str, Dict] = {}

# ---------- Funções de efeitos e ambientes (mantidas) ----------
def preload_all_effects():
    if not EFFECTS_DIR.exists():
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
    voice_entry = voices_registry.get(voice_name)
    if voice_entry:
        voice_dir = voice_entry["path"]
        effect_path = voice_dir / effect_file
        if effect_path.exists():
            try:
                seg = AudioSegment.from_wav(str(effect_path))
                if seg.frame_rate != SAMPLE_RATE_TARGET:
                    seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
                return seg
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar efeito '{effect_file}' da voz: {e}")
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

# ---------- Função que executa o binário piper (chamada no processo GPU) ----------
def synthesize_with_piper(voice_name: str, text: str, model_path: str, config_path: str,
                          speed: float, noise_scale: float, noise_w: float) -> bytes:
    """
    Executa o binário piper com aceleração CUDA e devolve os bytes WAV.
    """
    # Usa um ficheiro temporário em RAM (tmpfs) para máxima velocidade
    with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm", delete=False) as tmp:
        output_path = tmp.name

    try:
        cmd = [
            PIPER_BIN,
            "--model", model_path,
            "--config", config_path,
            "--output_file", output_path,
            "--cuda",
            "--length-scale", str(speed),
            "--noise-scale", str(noise_scale),
            "--noise-w", str(noise_w),
            "--text", text
        ]
        logger.info(f"Executando piper: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)

        # Lê o WAV gerado
        with open(output_path, "rb") as f:
            wav_data = f.read()
        return wav_data
    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass

def synthesize_to_audiosegment(voice_name: str, text: str, model_path: str, config_path: str,
                               speed: float, noise_scale: float, noise_w: float) -> AudioSegment:
    """Wrapper que devolve AudioSegment."""
    wav_bytes = synthesize_with_piper(voice_name, text, model_path, config_path,
                                      speed, noise_scale, noise_w)
    seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    if seg.frame_rate != SAMPLE_RATE_TARGET:
        seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
    return seg

# ---------- Registro de vozes (apenas metadados) ----------
def load_voice_metadata(voice_name: str, voice_path: Path) -> dict:
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
    return {
        "model_path": model_path,
        "config_path": config_path,
        "genero": genero,
        "path": voice_path
    }

def load_all_voices_metadata():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            name = item.name
            try:
                voices_registry[name] = load_voice_metadata(name, item)
                logger.info(f"✅ Metadados carregados: {name}")
            except Exception as e:
                logger.error(f"❌ Falha ao carregar metadados de {name}: {e}")
    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name not in voices_registry:
            json_file = onnx_file.with_suffix(".onnx.json")
            if json_file.exists():
                voices_registry[name] = {
                    "model_path": str(onnx_file),
                    "config_path": str(json_file),
                    "genero": "Personalizada",
                    "path": VOICES_DIR
                }
                logger.info(f"✅ Metadados personalizados: {name}")
    logger.info(f"Total de vozes disponíveis: {len(voices_registry)}")

# ---------- Mixagem ----------
def mix_and_export(segments: List[AudioSegment], ambient_cfg) -> bytes:
    if not segments:
        raise ValueError("Nenhum segmento")
    combined = AudioSegment.empty()
    for seg in segments:
        combined += seg

    target_dbfs = -20.0
    if combined.dBFS != target_dbfs:
        gain = target_dbfs - combined.dBFS
        combined = combined.apply_gain(gain)

    if ambient_cfg.enabled and ambient_cfg.file:
        ambient = get_ambient(ambient_cfg.file, ambient_cfg.volume_db)
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
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

app = FastAPI(title="Piper TTS API (GPU via binário)")

@app.on_event("startup")
async def startup():
    global gpu_pool, mix_pool, gpu_semaphore

    logger.info("Pré-carregando efeitos e ambientes...")
    preload_all_effects()
    preload_all_ambient()
    load_all_voices_metadata()

    # Pool de processos para GPU (1 processo dedicado)
    gpu_pool = ProcessPoolExecutor(max_workers=GPU_WORKERS)
    # Pool de threads para mixagem
    mix_pool = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    gpu_semaphore = asyncio.Semaphore(MAX_GPU_JOBS)

    logger.info("Sistema pronto (GPU via binário piper).")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    try:
        inicio = time.perf_counter()
        logger.info(f"🔊 Nova requisição: text='{req.text[:50]}...'")

        # ----- Mapeamento de speakers (igual ao original) -----
        is_dialog = bool(req.speakers)
        if not is_dialog:
            if not req.voice:
                raise HTTPException(400, "Campo 'voice' é obrigatório no modo simples")
            if req.voice not in voices_registry:
                raise HTTPException(404, f"Voz não encontrada: {req.voice}")
            speaker_params = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        else:
            speaker_params = {}
            for spk in req.speakers:
                noise_s = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
                noise_w = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
                speaker_params[spk.role] = (spk.voice, spk.speed, noise_s, noise_w)
            for role, (voice_name, _, _, _) in speaker_params.items():
                if voice_name not in voices_registry:
                    raise HTTPException(404, f"Voz '{voice_name}' do speaker '{role}' não encontrada")

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
                effect_file = req.effects[part]
                voice_for_effect = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                segments.append({"type": "effect", "effect_file": effect_file, "voice_name": voice_for_effect})
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

        async def process_segment(index: int, seg: dict) -> Tuple[int, AudioSegment]:
            if seg["type"] == "effect":
                return index, get_effect(seg["voice_name"], seg["effect_file"])

            voice_name = seg["voice_name"]
            meta = voices_registry[voice_name]

            async with gpu_semaphore:
                # Executa o binário piper no processo GPU
                audio = await loop.run_in_executor(
                    gpu_pool,
                    synthesize_to_audiosegment,
                    voice_name,
                    seg["text"],
                    meta["model_path"],
                    meta["config_path"],
                    seg["speed"],
                    seg["noise_s"],
                    seg["noise_w"]
                )
            return index, audio

        tasks = [process_segment(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        audio_segments = [seg for _, seg in results]

        final_wav = await loop.run_in_executor(
            mix_pool,
            mix_and_export,
            audio_segments,
            req.ambient
        )

        duracao_total = len(final_wav) / (2 * SAMPLE_RATE_TARGET)
        tempo_total = time.perf_counter() - inicio
        logger.info(f"✅ Síntese finalizada | tempo_total={tempo_total:.3f}s | áudio={duracao_total:.2f}s | RTF={tempo_total/duracao_total:.3f}")

        return Response(content=final_wav, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception:
        import traceback
        logger.error("Erro não tratado na síntese:\n" + traceback.format_exc())
        raise HTTPException(500, "Erro interno no processamento da síntese")

# Endpoints de saúde
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if len(voices_registry) > 0:
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
        "voices_loaded": list(voices_registry.keys()),
        "total_voices": len(voices_registry)
    }

@app.get("/gpu-health")
async def gpu_health():
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        smi = str(e)
    return {
        "nvidia_smi": smi.strip(),
        "binario": PIPER_BIN,
        "cuda_disponivel": os.path.exists(PIPER_BIN)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
