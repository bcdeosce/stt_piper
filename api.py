#!/ docs
import os
import re
import io
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Piper imports (da venv)
from piper import PiperVoice
from piper.download import ensure_voice_exists, get_voices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-api")

BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Carrega vozes (sem pool, usamos sob demanda com lock por voz) ----------
voice_locks: Dict[str, any] = {}  # Para evitar concorrência, usaremos um lock simples por voz
voices_models: Dict[str, dict] = {}  # armazena modelo_path e config_path

# Função para carregar uma voz (sem instanciar PiperVoice ainda)
def load_voice_metadata(voice_name: str, voice_path: Path):
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
    voices_models[voice_name] = {"model_path": model_path, "config_path": config_path}
    voice_locks[voice_name] = threading.Lock()  # thread lock
    logger.info(f"✅ Metadados da voz {voice_name} carregados")

for item in VOICES_DIR.iterdir():
    if item.is_dir():
        try:
            load_voice_metadata(item.name, item)
        except Exception as e:
            logger.error(f"❌ {item.name}: {e}")

for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name not in voices_models:
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                load_voice_metadata(voice_name, VOICES_DIR)
            except Exception as e:
                logger.error(f"❌ {voice_name}: {e}")

logger.info(f"Total de vozes: {len(voices_models)}")

# ---------- Função de síntese (cria uma sessão por chamada, com lock para evitar concorrência) ----------
import threading

def synthesize_text(voice_name: str, text: str, length_scale: float, noise_scale: float, noise_w_scale: float):
    if voice_name not in voices_models:
        raise ValueError(f"Voz {voice_name} não encontrada")
    lock = voice_locks[voice_name]
    with lock:
        # Carrega o modelo (cached? na verdade, recarrega a cada vez; para performance, seria melhor cachear a sessão,
        # mas a sessão ONNX não é thread-safe. Para simplificar e evitar problemas de GPU, criamos uma nova a cada chamada.)
        voice = PiperVoice.load(
            voices_models[voice_name]["model_path"],
            config_path=voices_models[voice_name]["config_path"],
            use_cuda=True   # <--- GPU ATIVADA
        )
        # Sintetiza
        audio_stream = voice.synthesize(
            text,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w_scale
        )
        # Concatena os chunks de áudio
        pcm_bytes = b''.join(audio_chunk.audio_int16_bytes for audio_chunk in audio_stream)
        sample_rate = voice.config.sample_rate
        del voice  # libera memória GPU
    return pcm_bytes, sample_rate

# ---------- Modelos de requisição (mesmos de antes) ----------
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

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS GPU (Python)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...'")

    # Lógica de diálogo (idêntica à anterior)
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório")
        if req.voice not in voices_models:
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
            if v not in voices_models:
                raise HTTPException(404, f"Voz '{v}' do speaker '{role}' não encontrada")
        current_role = None

    parts = re.split(r'(\[.*?\])', req.text)
    audio_chunks = []

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
            # Efeitos (mesmo código)
            effect_name = req.effects[part]
            voice_name_eff = speaker_map[current_role][0] if is_dialog and current_role else req.voice
            effect_path = None
            voice_dir = Path(voices_models[voice_name_eff]["model_path"]).parent
            candidate = voice_dir / effect_name
            if candidate.exists(): effect_path = candidate
            if not effect_path: candidate = EFFECTS_DIR / effect_name
            if candidate.exists(): effect_path = candidate
            if not effect_path:
                raise HTTPException(404, f"Efeito '{effect_name}' não encontrado")
            seg = AudioSegment.from_wav(str(effect_path))
            if seg.frame_rate != 22050: seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
            continue

        # Fala
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Speaker não definido")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        try:
            pcm, sr = synthesize_text(voice_name, part, speed, noise_s, noise_w)
            seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
            if sr != 22050: seg = seg.set_frame_rate(22050)
            audio_chunks.append(seg)
        except Exception as e:
            logger.error(f"Erro na síntese: {e}")
            raise HTTPException(500, f"Erro na voz {voice_name}: {e}")

    if not audio_chunks:
        raise HTTPException(400, "Nenhum áudio gerado")

    combined = sum(audio_chunks, AudioSegment.empty())
    target_dBFS = -20.0
    if combined.dBFS != target_dBFS:
        combined = combined.apply_gain(target_dBFS - combined.dBFS)

    if req.ambient.enabled and req.ambient.file:
        ambient_path = AMBIENT_DIR / f"{req.ambient.file}.wav"
        if not ambient_path.exists():
            raise HTTPException(404, f"Ambiente não encontrado")
        ambient = AudioSegment.from_wav(str(ambient_path))
        if ambient.frame_rate != combined.frame_rate:
            ambient = ambient.set_frame_rate(combined.frame_rate)
        if len(ambient) < len(combined):
            ambient = ambient * (len(combined) // len(ambient) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)

    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
    dur = time.perf_counter() - inicio
    logger.info(f"✅ Finalizado em {dur:.2f}s, WebM de {buf.tell()} bytes")
    return Response(content=buf.getvalue(), media_type="audio/webm")

# Health checks
@app.get("/started")
async def started(): return Response(status_code=200, content="started")
@app.get("/ready")
async def ready():
    if voices_models: return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")
@app.get("/live")
async def live(): return Response(status_code=200, content="alive")
@app.get("/health")
async def health():
    return {"status": "ok", "gpu": True, "voices": list(voices_models.keys())}
