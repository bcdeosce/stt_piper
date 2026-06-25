import os
import re
import io
import wave
import time
import queue
import asyncio
import logging
import subprocess
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydub import AudioSegment

# ---------- Configuração de logs ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-api")

# ---------- Constantes ----------
MAX_GPU_JOBS = 3
GPU_WORKERS = 2
MIX_WORKERS = 10
SAMPLE_RATE_TARGET = 22050

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Referências atrasadas ----------
_onnxruntime = None

def get_ort():
    if _onnxruntime is None:
        raise RuntimeError("onnxruntime não inicializado")
    return _onnxruntime

# ---------- Tokens especiais ----------
BOS = "^"
EOS = "$"
PAD = "_"

def audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    max_int16 = np.iinfo(np.int16).max
    audio = np.clip(audio, -1, 1) * max_int16
    return audio.astype(np.int16)

# ---------- Classe de voz personalizada ----------
class CustomPiperVoice:
    def __init__(self, model_path: str, config_path: str, use_cuda: bool = True):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Acesso seguro às chaves (fallbacks para formatos antigos)
        self.sample_rate = cfg.get("audio", {}).get("sample_rate", cfg.get("sample_rate", 22050))
        self.phoneme_type = cfg.get("phoneme_type", "espeak")
        self.id_map = cfg["phoneme_id_map"]
        self.num_speakers = cfg.get("num_speakers", 1)
        self.length_scale = cfg.get("length_scale", 1.0)
        self.noise_scale = cfg.get("noise_scale", 0.667)
        self.noise_w = cfg.get("noise_w", 0.8)
        # Espeak pode estar dentro de "espeak" ou "espeak_voice"
        espeak_cfg = cfg.get("espeak", {})
        self.espeak_voice = espeak_cfg.get("voice") or cfg.get("espeak_voice", "en")

        ort = get_ort()
        providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=ort.SessionOptions(),
            providers=providers
        )

    def phonemize(self, text: str) -> List[List[str]]:
        from piper_phonemize import phonemize_espeak, phonemize_codepoints, tashkeel_run
        if self.phoneme_type == "espeak":
            if self.espeak_voice == "ar":
                text = tashkeel_run(text)
            return phonemize_espeak(text, self.espeak_voice)
        elif self.phoneme_type == "text":
            return phonemize_codepoints(text)
        else:
            raise ValueError(f"Tipo de fonema inesperado: {self.phoneme_type}")

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        ids = [self.id_map[BOS]]
        for p in phonemes:
            if p not in self.id_map:
                logger.warning("Fonema em falta no id_map: %s", p)
                continue
            ids.extend(self.id_map[p])
            ids.append(self.id_map[PAD])
        ids.append(self.id_map[EOS])
        return ids

    def synthesize_ids_raw(
        self,
        phoneme_ids: List[int],
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        length_scale = length_scale if length_scale is not None else self.length_scale
        noise_scale = noise_scale if noise_scale is not None else self.noise_scale
        noise_w = noise_w if noise_w is not None else self.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        sid = None
        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)

        audio = self.session.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())
        return audio.tobytes()

    def synthesize(
        self,
        text: str,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
    ) -> Iterable[bytes]:
        sentence_phonemes = self.phonemize(text)
        num_silence_samples = int(sentence_silence * self.sample_rate)
        silence_bytes = bytes(num_silence_samples * 2)

        for phonemes in sentence_phonemes:
            ids = self.phonemes_to_ids(phonemes)
            audio_chunk = self.synthesize_ids_raw(
                ids,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
            yield audio_chunk + silence_bytes

    def synthesize_to_audiosegment(
        self,
        text: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.0,
    ) -> AudioSegment:
        raw_bytes = b''.join(self.synthesize(
            text,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
            sentence_silence=sentence_silence,
        ))
        seg = AudioSegment(
            data=raw_bytes,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=1
        )
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        return seg

# ---------- Pools e registo ----------
voices_registry: Dict = {}
EFFECTS_CACHE: Dict[str, AudioSegment] = {}
AMBIENT_CACHE: Dict[str, AudioSegment] = {}

class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = CustomPiperVoice(model_path, config_path, use_cuda=True)
            self.pool.put(voice)

    def get(self, timeout: float = 2.0):
        return self.pool.get(timeout=timeout)

    def put(self, voice):
        self.pool.put(voice)

# ---------- Funções de carregamento de vozes e caches (mantidas iguais) ----------
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
    # ... (exatamente como nas versões anteriores, sem alterações)
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            voice_name = item.name
            try:
                entry = load_voice_from_folder(voice_name, item)
                voices_registry[voice_name] = entry
                logger.info(f"✅ Voz carregada: {voice_name} ({entry['genero']})")
            except Exception as e:
                logger.error(f"❌ Falha ao carregar voz {voice_name}: {e}")

    for onnx_file in VOICES_DIR.glob("*.onnx"):
        voice_name = onnx_file.stem
        if voice_name in voices_registry:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                pool = VoicePool(str(onnx_file), str(json_file), pool_size=2)
                voices_registry[voice_name] = {
                    "model_path": str(onnx_file),
                    "config_path": str(json_file),
                    "genero": "Personalizada",
                    "pool": pool,
                    "path": VOICES_DIR
                }
                logger.info(f"✅ Voz personalizada (raiz) carregada: {voice_name}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar voz {voice_name}: {e}")

    logger.info(f"Total de vozes disponíveis: {len(voices_registry)}")

def preload_all_effects():
    # ... (igual)
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
    # ... (igual)
    if not AMBIENT_DIR.exists():
        return
    for wav_file in AMBIENT_DIR.glob("*.wav"):
        try:
            seg = AudioSegment.from_wav(str(wav_file))
            if seg.frame_rate != SAMPLE_RATE_TARGET:
                seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
            base_name = wav_file.stem
            AMBIENT_CACHE[base_name] = seg
            logger.info(f"✔ Ambiente pré-carregado: {base_name}")
        except Exception as e:
            logger.error(f"Erro ao carregar ambiente {wav_file.name}: {e}")

def get_effect(voice_name: str, effect_file: str) -> AudioSegment:
    # ... (igual)
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
    # ... (igual)
    base_name = ambient_file
    if base_name not in AMBIENT_CACHE:
        ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
        if not ambient_path.exists():
            raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")
        seg = AudioSegment.from_wav(str(ambient_path))
        if seg.frame_rate != SAMPLE_RATE_TARGET:
            seg = seg.set_frame_rate(SAMPLE_RATE_TARGET)
        AMBIENT_CACHE[base_name] = seg
    return AMBIENT_CACHE[base_name] + volume_db

def synthesize_speech(voice: CustomPiperVoice, text: str, speed: float,
                      noise_s: float, noise_w: float) -> AudioSegment:
    return voice.synthesize_to_audiosegment(
        text,
        length_scale=speed,
        noise_scale=noise_s,
        noise_w=noise_w
    )

def mix_and_export(segments: List[AudioSegment],
                   ambient_cfg) -> bytes:
    # ... (igual)
    if not segments:
        raise ValueError("Nenhum segmento para mixar")
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

# ---------- Modelos ----------
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
    voice: Optional[str] = Field(None, description="Nome da voz (modo único)")
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- App ----------
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Piper TTS API (GPU factory)")

gpu_executor: Optional[ThreadPoolExecutor] = None
mix_executor: Optional[ThreadPoolExecutor] = None
gpu_semaphore: asyncio.Semaphore = None

# ---------- Instalação dinâmica ----------
def install_dependencies():
    try:
        for f in Path("/app/lib").glob("libonnxruntime*"):
            f.unlink()
            logger.info(f"Biblioteca antiga removida: {f}")
    except Exception as e:
        logger.warning(f"Erro ao remover libs antigas: {e}")

    logger.info("Instalando numpy e onnxruntime-gpu...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "numpy==1.26.4", "onnxruntime-gpu==1.18.0"],
            stdout=sys.stdout, stderr=sys.stderr
        )
        logger.info("Instalação concluída com sucesso.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Falha na instalação: {e}")
        raise

def diagnose_environment():
    ort = get_ort()
    logger.info("========== DIAGNÓSTICO DO AMBIENTE ==========")
    logger.info(f"onnxruntime versão: {ort.__version__}")
    providers = ort.get_available_providers()
    logger.info(f"Providers disponíveis: {providers}")
    logger.info(f"Dispositivo padrão: {ort.get_device()}")

    try:
        smi_out = subprocess.check_output(["nvidia-smi"], text=True)
        logger.info("nvidia-smi:\n" + smi_out.strip())
    except Exception as e:
        logger.warning(f"nvidia-smi não disponível: {e}")

    for var in ["LD_LIBRARY_PATH", "CUDA_HOME", "PATH", "NVIDIA_VISIBLE_DEVICES"]:
        logger.info(f"{var}={os.environ.get(var, 'não definida')}")

    if 'CUDAExecutionProvider' in providers:
        logger.info("CUDAExecutionProvider está disponível e será utilizado.")
    else:
        logger.warning("CUDAExecutionProvider NÃO está na lista de providers!")
    logger.info("=============================================")

@app.on_event("startup")
async def startup_event():
    global _onnxruntime
    global gpu_executor, mix_executor, gpu_semaphore

    logger.info("Inicializando dependências e ambiente...")
    install_dependencies()

    import onnxruntime as _ort_mod
    _onnxruntime = _ort_mod

    diagnose_environment()
    preload_all_effects()
    preload_all_ambient()
    load_all_voices()

    gpu_executor = ThreadPoolExecutor(max_workers=GPU_WORKERS)
    mix_executor = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    gpu_semaphore = asyncio.Semaphore(MAX_GPU_JOBS)

    logger.info("Sistema pronto.")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    try:
        inicio_total = time.perf_counter()
        logger.info(f"🔊 Nova requisição: text='{req.text[:50]}...', effects={list(req.effects.keys())}, ambient={req.ambient.enabled}")

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
            pool = voices_registry[voice_name]["pool"]

            async with gpu_semaphore:
                voice = await loop.run_in_executor(gpu_executor, pool.get)
                try:
                    audio = await loop.run_in_executor(
                        gpu_executor,
                        synthesize_speech,
                        voice,
                        seg["text"],
                        seg["speed"],
                        seg["noise_s"],
                        seg["noise_w"]
                    )
                finally:
                    pool.put(voice)
            return index, audio

        tasks = [process_segment(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        audio_segments = [seg for _, seg in results]

        final_wav = await loop.run_in_executor(
            mix_executor,
            mix_and_export,
            audio_segments,
            req.ambient
        )

        duracao_total = len(final_wav) / (2 * SAMPLE_RATE_TARGET)
        tempo_total = time.perf_counter() - inicio_total
        logger.info(f"✅ Síntese finalizada | tempo_total={tempo_total:.3f}s | áudio={duracao_total:.2f}s | RTF={tempo_total/duracao_total:.3f}")

        return Response(content=final_wav, media_type="audio/wav")

    except HTTPException:
        raise  # deixa o FastAPI lidar com respostas HTTP apropriadas
    except Exception:
        logger.error("Erro não tratado na síntese:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno no processamento da síntese")

# ---------- Endpoints de saúde ----------
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
    ort = get_ort()
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
        "env": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
