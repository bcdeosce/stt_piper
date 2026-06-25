#!/usr/bin/env python3
"""
Piper TTS API – usando biblioteca piper-tts com suporte a GPU (onnxruntime-gpu)
Instalação automática do pacote com [gpu] se ausente.
Logs completos, sem supressão.
"""

import os
import sys
import subprocess
import importlib

# ----------------------------------------------------------------------
# 1. GARANTIR QUE O PACOTE piper-tts[gpu] ESTEJA INSTALADO
# ----------------------------------------------------------------------
def ensure_piper_installed():
    try:
        importlib.import_module("piper")
        # Verifica se onnxruntime-gpu está instalado
        try:
            importlib.import_module("onnxruntime")
            # Se veio da versão gpu, ok; senão, tenta instalar gpu
            import onnxruntime
            if not hasattr(onnxruntime, 'get_device') or onnxruntime.get_device() != 'GPU':
                print("⚠️  onnxruntime encontrado, mas não parece ser GPU. Tentando atualizar...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "onnxruntime-gpu", "--upgrade"
                ])
        except ImportError:
            print("⚙️  onnxruntime não encontrado. Instalando onnxruntime-gpu...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "onnxruntime-gpu"
            ])
        print("✅ Piper com suporte GPU já instalado.")
    except ImportError:
        print("📦 Piper não encontrado. Instalando piper-tts[gpu]...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "piper-tts[gpu]"
        ])
        print("✅ Piper instalado com suporte GPU.")

ensure_piper_installed()

# ----------------------------------------------------------------------
# 2. IMPORTAÇÕES (agora com pacotes disponíveis)
# ----------------------------------------------------------------------
import re
import io
import json
import time
import queue
import logging
import asyncio
import threading
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment
import numpy as np
from piper import PiperVoice

# Configuração de logging (visível, sem supressão)
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

# Diagnóstico do ambiente (inclui nvidia-smi)
logger.info("=" * 70)
logger.info("DIAGNÓSTICO DE AMBIENTE - GPU")
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
# PiperVoicePool – usando a biblioteca Python com GPU
# =============================================================================
class PiperVoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = PIPER_POOL_SIZE):
        self.model_path = model_path
        self.pool = queue.Queue(maxsize=pool_size)
        # Uso de GPU controlado por variável de ambiente (padrão: true)
        use_cuda_env = os.getenv("PIPER_USE_CUDA", "true").lower()
        self.use_cuda = use_cuda_env in ("true", "1", "yes")
        logger.info(f"Carregando voz {Path(model_path).stem} com CUDA={self.use_cuda}")
        for i in range(pool_size):
            try:
                voice = PiperVoice.load(model_path, use_cuda=self.use_cuda)
                self.pool.put(voice)
                logger.info(f"  → Worker {i+1}/{pool_size} carregado (GPU={self.use_cuda})")
            except Exception as e:
                logger.error(f"  ✗ Falha ao criar worker {i+1}: {e}")
        if self.pool.empty():
            raise RuntimeError("Nenhum worker Piper iniciado.")

    def synthesize(self, text: str, length_scale: float = 1.0,
                   noise_scale: float = 0.667, noise_w_scale: float = 0.8) -> Tuple[bytes, int]:
        """
        Retorna (áudio PCM em bytes, sample_rate)
        """
        voice = self.pool.get()
        try:
            audio_array, sample_rate = voice.synthesize(
                text,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w_scale
            )
            # Converte numpy array (int16) para bytes
            audio_bytes = audio_array.astype(np.int16).tobytes()
            return audio_bytes, sample_rate
        finally:
            self.pool.put(voice)

# =============================================================================
# CARREGAMENTO DAS VOZES
# =============================================================================
voice_pools: Dict[str, PiperVoicePool] = {}

def load_voice(voice_name: str, voice_dir: Path):
    onnx_files = list(voice_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_dir}")
    model_path = str(onnx_files[0])
    logger.info(f"Carregando voz {voice_name} (pool de {PIPER_POOL_SIZE})")
    pool = PiperVoicePool(model_path, "", pool_size=PIPER_POOL_SIZE)
    voice_pools[voice_name] = pool
    logger.info(f"✅ Voz {voice_name} pronta (GPU)")

# Carrega vozes em subdiretórios
for item in VOICES_DIR.iterdir():
    if item.is_dir():
        try:
            load_voice(item.name, item)
        except Exception as e:
            logger.error(f"❌ {item.name}: {e}")

# Carrega vozes diretamente na raiz VOICES_DIR
for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voice_pools:
        continue
    try:
        load_voice(voice_name, VOICES_DIR)
    except Exception as e:
        logger.error(f"❌ {voice_name}: {e}")

logger.info(f"Total de vozes carregadas: {len(voice_pools)}")
MODEL_LOADED = len(voice_pools) > 0

# =============================================================================
# WARM-UP (teste com voz 'faber' se disponível)
# =============================================================================
logger.info("=" * 70)
logger.info("🔥 WARM-UP")
warmup_success = False
if "faber" in voice_pools:
    try:
        pcm, sr = voice_pools["faber"].synthesize("Teste de aquecimento", 1.0, 0.667, 0.8)
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
        if sr != 22050:
            seg = seg.set_frame_rate(22050)
        buf = io.BytesIO()
        seg.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
        warmup_success = True
        logger.info(f"✅ Warm-up OK – WebM de {buf.tell()} bytes")
    except Exception as e:
        logger.critical(f"❌ Warm-up falhou: {e}")
else:
    logger.warning("Voz 'faber' não encontrada, warm-up ignorado.")
logger.info("=" * 70)

# =============================================================================
# MIXAGEM (processo separado, sem alterações)
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
    if not audio_chunks:
        raise ValueError("Nenhum áudio")
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
        combined = combined.overlay(ambient, gain_during_overlay=ambient_volume_db)
    buf = io.BytesIO()
    combined.export(buf, format="webm", codec="libopus", parameters=["-b:a", "64k"])
    log.info(f"Mixagem concluída: {buf.tell()} bytes")
    return buf.getvalue()

# =============================================================================
# MODELOS PYDANTIC (sem alterações)
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
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="Piper TTS (biblioteca Python + GPU)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    inicio = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Nova requisição: '{req.text[:80]}...'")

    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório")
        if req.voice not in voice_pools:
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
            if v not in voice_pools:
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
        # Se for marcador de speaker (ex: [joao])
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue
        # Se for efeito (ex: {tosse})
        if part in req.effects:
            effect_name = req.effects[part]
            if effect_name not in effect_cache:
                voice_name_eff = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                voice_dir = Path(voice_pools[voice_name_eff].pool.queue[0].model_path).parent
                effect_path = None
                if (voice_dir / effect_name).exists():
                    effect_path = voice_dir / effect_name
                else:
                    candidate = EFFECTS_DIR / effect_name
                    if candidate.exists():
                        effect_path = candidate
                if not effect_path:
                    raise HTTPException(404, f"Efeito '{effect_name}' não encontrado")
                with open(effect_path, "rb") as f:
                    effect_cache[effect_name] = f.read()
            ordered_items.append({'type': 'effect', 'wav_bytes': effect_cache[effect_name]})
            continue

        # Texto comum (ou fala)
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Speaker não definido para o texto")
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

    # Executa síntese em paralelo
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

    # Carrega áudio ambiente se ativado
    ambient_bytes = None
    if req.ambient.enabled and req.ambient.file:
        ambient_path = AMBIENT_DIR / f"{req.ambient.file}.wav"
        if not ambient_path.exists():
            raise HTTPException(404, f"Ambiente '{req.ambient.file}' não encontrado")
        with open(ambient_path, "rb") as f:
            ambient_bytes = f.read()

    # Mixagem em processo separado
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

# =============================================================================
# HEALTH CHECKS
# =============================================================================
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
    return {
        "status": "ok",
        "gpu": True,
        "warmup": warmup_success,
        "voices_loaded": list(voice_pools.keys()),
        "total_voices": len(voice_pools)
    }

# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
