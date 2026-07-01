import os
import re
import sys
import time
import json
import logging
import subprocess
import threading
import asyncio
import io
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict, deque

# ========== INSTALAR PSUTIL SE NÃO EXISTIR ==========
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# ========== FORÇA 1 THREAD NO ONNX RUNTIME ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort

_original_ort_session = ort.InferenceSession
def _patched_ort_session(model_path, sess_options=None, providers=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return _original_ort_session(model_path, sess_options, providers=providers, **kwargs)
ort.InferenceSession = _patched_ort_session

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
    from piper import PiperVoice, SynthesisConfig

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ort.set_default_logger_severity(4)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

class MemoryHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.capacity = capacity
        self.buffer = []
    def emit(self, record):
        self.buffer.append(self.format(record))
        if len(self.buffer) > self.capacity: self.buffer.pop(0)

memory_handler = MemoryHandler(capacity=100)
memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), memory_handler])
logging.getLogger().handlers[0].setLevel(LOG_LEVEL)
memory_handler.setLevel(logging.DEBUG)
logger = logging.getLogger("piper-api")
logger.setLevel(logging.DEBUG)

# ---------- Variáveis de ambiente ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", "6"))
CPU_WORKERS = int(os.getenv("CPU_WORKERS", "3"))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", "2"))

POOL_INIT_SIZE = int(os.getenv("POOL_INIT_SIZE", "1"))
POOL_MAX_SIZE  = int(os.getenv("POOL_MAX_SIZE", str(TTS_WORKERS)))
POOL_EXTRA_TTL_MINUTES = float(os.getenv("POOL_EXTRA_TTL_MINUTES", "15"))
POOL_RESET_MINUTES = float(os.getenv("POOL_RESET_MINUTES", "0"))

SAMPLE_RATE_TARGET = 22050
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
BENCH_DIR = BASE_DIR / "bench"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)
BENCH_DIR.mkdir(exist_ok=True)
CARGA_FILE = BENCH_DIR / "latest_carga.json"

# ---------- Benchmarks ----------
tts_wall_times, mix_times, queue_wait_times, total_times = [], [], [], []
bench_lock = threading.Lock()
worker_metrics = defaultdict(lambda: {"count": 0, "total_synth_time": 0.0, "jobs": []})
worker_metrics_lock = threading.Lock()
_active_synthesis_count = 0
_active_lock = threading.Lock()
stats = defaultdict(list)
stats_lock = threading.Lock()

def compute_stats(values):
    if not values: return {"mean":0,"min":0,"max":0,"p95":0,"count":0}
    arr = np.array(values)
    return {"mean":float(np.mean(arr)),"min":float(np.min(arr)),"max":float(np.max(arr)),"p95":float(np.percentile(arr,95)),"count":len(values)}

# ====================== DETEÇÃO DE NÚCLEOS (ROBUSTA) ======================
def _get_cores():
    try:
        physical_logical, ht_logical = set(), set()
        total = os.cpu_count() or 4
        for i in range(total):
            try:
                with open(f'/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list') as f:
                    sibs = list(map(int, f.read().strip().split(',')))
                    if len(sibs) > 1 and i != sibs[0]:
                        ht_logical.add(i)
                    else:
                        physical_logical.add(i)
            except:
                if i < total // 2:
                    physical_logical.add(i)
                else:
                    ht_logical.add(i)
        if physical_logical and ht_logical:
            return sorted(physical_logical), sorted(ht_logical)
    except Exception as e:
        logger.warning(f"Falha na deteção fina de núcleos: {e}")
    total = os.cpu_count() or 4
    return list(range(total // 2)), list(range(total // 2, total))

_PHYSICAL, _HT = _get_cores()
def _parse_cores(var, default):
    val = os.getenv(var, "")
    if val: return [int(x.strip()) for x in val.split(",")]
    return default

TTS_CORES = _parse_cores("TTS_CORES", _HT[:TTS_WORKERS])
CPU_CORES = _parse_cores("CPU_CORES", _HT[TTS_WORKERS:TTS_WORKERS+CPU_WORKERS])
MIX_CORES = _parse_cores("MIX_CORES", _PHYSICAL[:MIX_WORKERS])

logger.info(f"Núcleos físicos detetados: {_PHYSICAL}, HT: {_HT}")
logger.info(f"TTS_CORES (HT): {TTS_CORES}, CPU_CORES (HT): {CPU_CORES}, MIX_CORES (físicos): {MIX_CORES}")

# ====================== POOL GPU ======================
_manager = mp.Manager()
_tts_core_queue = _manager.Queue()
for core in TTS_CORES: _tts_core_queue.put(core)

def _init_tts_worker():
    core = _tts_core_queue.get()
    try: os.sched_setaffinity(0, {core})
    except Exception as e: logger.warning(f"Falha ao fixar worker TTS no núcleo {core}: {e}")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    import onnxruntime as _ort
    _orig = _ort.InferenceSession
    def _patch(model_path, sess_options=None, providers=None, **kwargs):
        if sess_options is None: sess_options = _ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return _orig(model_path, sess_options, providers=providers, **kwargs)
    _ort.InferenceSession = _patch
    import __main__
    __main__._worker_voice_cache = {}

def _synthesize_text_worker(voice_name, phoneme_ids, scales_tuple):
    cache = getattr(sys.modules['__main__'], '_worker_voice_cache', {})
    if voice_name not in cache:
        model_path, config_path = _get_voice_paths(voice_name)
        logger.debug(f"[GPU worker {os.getpid()}] Carregando voz {voice_name}")
        voice = PiperVoice.load(model_path, config_path, use_cuda=True)
        cache[voice_name] = voice
    voice = cache[voice_name]
    if not phoneme_ids:
        logger.error(f"[GPU worker {os.getpid()}] phoneme_ids vazio para voz {voice_name}")
        raise ValueError("phoneme_ids está vazio")
    length_scale, noise_scale, noise_w = scales_tuple
    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
    scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
    args = {"input": phoneme_ids_array, "input_lengths": phoneme_ids_lengths, "scales": scales}
    if voice.config.num_speakers > 1: args["sid"] = np.array([0], dtype=np.int64)
    t0 = time.perf_counter()
    audio = voice.session.run(None, args)[0].squeeze()
    audio = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
    synth_time = time.perf_counter() - t0
    logger.debug(f"[GPU worker {os.getpid()}] Síntese OK em {synth_time:.3f}s, amostras={len(audio)}")
    return voice.config.sample_rate, audio.tobytes(), synth_time, os.getpid()

_VOICE_PATHS: Dict[str, Tuple[str, str]] = {}
def _get_voice_paths(voice_name):
    if voice_name not in _VOICE_PATHS:
        vp = VOICES_DIR / voice_name
        onnx_files = list(vp.glob("*.onnx"))
        if not onnx_files: raise FileNotFoundError(f"Nenhum .onnx em {vp}")
        mp_path = str(onnx_files[0])
        base = onnx_files[0].stem
        jp = vp / f"{base}.onnx.json"
        if not jp.exists(): jp = next(vp.glob("*.json"))
        _VOICE_PATHS[voice_name] = (mp_path, str(jp))
        logger.debug(f"Registrada voz {voice_name}: model={mp_path}, config={jp}")
    return _VOICE_PATHS[voice_name]

# ---------- Registro de vozes ----------
voices_registry: Dict[str, Dict] = {}
def load_voice_metadata(voice_name, voice_path):
    onnx_files = list(voice_path.glob("*.onnx"))
    mp_path = str(onnx_files[0])
    base = onnx_files[0].stem
    jp = voice_path / f"{base}.onnx.json"
    if not jp.exists(): jp = next(voice_path.glob("*.json"))
    _VOICE_PATHS[voice_name] = (mp_path, str(jp))
    gen = "Desconhecido"
    meta = voice_path / f"{voice_name}.json"
    if meta.exists():
        try:
            with open(meta) as f: gen = json.load(f).get("genero","Desconhecido")
        except: pass
    logger.debug(f"Metadados registrados para voz {voice_name}")
    return {"model_path":mp_path, "config_path":str(jp), "genero":gen, "path":voice_path}

def load_all_voices_metadata():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            try: voices_registry[item.name] = load_voice_metadata(item.name, item)
            except Exception as e: logger.error(f"Falha ao registar {item.name}: {e}")
    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name in voices_registry: continue
        jf = onnx_file.with_suffix(".onnx.json")
        if jf.exists():
            try:
                _VOICE_PATHS[name] = (str(onnx_file), str(jf))
                voices_registry[name] = {"model_path":str(onnx_file),"config_path":str(jf),"genero":"Personalizada","path":VOICES_DIR}
            except: pass
    logger.info(f"Total de vozes registadas: {len(voices_registry)}")

# ---------- Cache de instâncias CPU para fonemização ----------
_cpu_voice_cache = {}
_cpu_cache_lock = threading.Lock()

def phonemize_and_ids(voice_name, text, speed, noise_scale, noise_w_scale):
    start = time.perf_counter()
    with _cpu_cache_lock:
        if voice_name not in _cpu_voice_cache:
            mp_path, cp_path = _get_voice_paths(voice_name)
            logger.debug(f"[CPU fonemização] Carregando voz {voice_name} (CPU)")
            _cpu_voice_cache[voice_name] = PiperVoice.load(mp_path, cp_path, use_cuda=False)
        voice = _cpu_voice_cache[voice_name]
    logger.debug(f"[CPU fonemização] Processando texto: '{text[:60]}...'")
    try:
        phonemes = voice.phonemize(text)
        if not phonemes or not phonemes[0]:
            logger.warning(f"[CPU fonemização] Fonemização vazia para texto: '{text[:50]}...'")
            return voice_name, [], (speed, noise_scale, noise_w_scale)
        phoneme_ids = voice.phonemes_to_ids(phonemes[0])
        elapsed = time.perf_counter() - start
        logger.debug(f"[CPU fonemização] OK em {elapsed*1000:.1f}ms, {len(phoneme_ids)} IDs")
        return voice_name, phoneme_ids, (speed, noise_scale, noise_w_scale)
    except Exception as e:
        logger.error(f"[CPU fonemização] ERRO: {e} | texto: '{text[:80]}'")
        return voice_name, [], (speed, noise_scale, noise_w_scale)

# ---------- Mixagem ----------
def mix_and_concat(segments_data, ambient_cfg, target_rate=22050):
    t_start = time.perf_counter()
    combined = AudioSegment.empty()
    for data in segments_data:
        if 'pcm_bytes' in data:
            seg = AudioSegment(data=data['pcm_bytes'], sample_width=2, frame_rate=data['sample_rate'], channels=1)
            if seg.frame_rate != target_rate: seg = seg.set_frame_rate(target_rate)
            combined += seg
        elif 'effect' in data:
            vdir = voices_registry[data['voice']]["path"]
            epath = vdir / data['effect']
            if not epath.exists(): epath = EFFECTS_DIR / data['effect']
            if epath.exists():
                try:
                    eseg = AudioSegment.from_wav(str(epath))
                    if eseg.frame_rate != target_rate: eseg = eseg.set_frame_rate(target_rate)
                    combined += eseg
                except Exception as e: logger.error(f"Erro ao carregar efeito '{data['effect']}': {e}")
            else:
                logger.debug(f"Efeito '{data['effect']}' não encontrado")
    if len(combined) == 0: raise RuntimeError("Nenhum áudio foi gerado.")
    target_dbfs = -20.0
    if combined.dBFS != target_dbfs: combined = combined.apply_gain(target_dbfs - combined.dBFS)
    if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
        ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
        if ambient_path.exists():
            try:
                ambient = AudioSegment.from_wav(str(ambient_path))
                if ambient.frame_rate != target_rate: ambient = ambient.set_frame_rate(target_rate)
                ambient = ambient + ambient_cfg.get('volume_db', -15.0)
                if len(ambient) < len(combined): ambient = ambient * ((len(combined)//len(ambient)) + 1)
                ambient = ambient[:len(combined)]
                combined = combined.overlay(ambient)
                logger.debug(f"Ambiente '{ambient_cfg['file']}' aplicado")
            except Exception as e: logger.error(f"Erro ao aplicar ambiente: {e}")
        else:
            logger.debug(f"Ficheiro de ambiente '{ambient_cfg['file']}.wav' não encontrado")
    with io.BytesIO() as buf:
        combined.export(buf, format="wav")
        wav_bytes = buf.getvalue()
    elapsed = time.perf_counter() - t_start
    logger.debug(f"Mixagem concluída em {elapsed*1000:.1f}ms")
    return wav_bytes, elapsed

# ---------- Modelos ----------
class AmbientConfig(BaseModel):
    enabled: bool = False; file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)
class SpeakerMapping(BaseModel):
    role: str; voice: str; speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)
class TTSRequest(BaseModel):
    voice: Optional[str] = Field(None); text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS API (GPU)")
tts_pool: Optional[ProcessPoolExecutor] = None
cpu_pool: Optional[ThreadPoolExecutor] = None
mix_pool: Optional[ThreadPoolExecutor] = None

@app.on_event("startup")
async def startup():
    global tts_pool, cpu_pool, mix_pool
    load_all_voices_metadata()
    tts_pool = ProcessPoolExecutor(max_workers=TTS_WORKERS, initializer=_init_tts_worker)
    cpu_pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)
    mix_pool = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    logger.info(f"Sistema pronto. TTS={TTS_WORKERS}, CPU={CPU_WORKERS}, MIX={MIX_WORKERS}")

# ================= ENDPOINT DE SÍNTESE =================
@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()
    logger.debug(f"Requisição recebida: '{req.text[:60]}...'")
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
    segments_data = []
    loop = asyncio.get_running_loop()

    async def process_part(part):
        nonlocal current_role
        part = part.strip()
        if not part: return None
        if part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if is_dialog and role in speaker_params:
                current_role = role
                return None
            if part in req.effects:
                effect_file = req.effects[part]
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                logger.debug(f"Efeito '{effect_file}' agendado para voz '{vname}'")
                return {'effect': effect_file, 'voice': vname}
            return None
        if is_dialog:
            if current_role is None: raise HTTPException(400, "Speaker não definido")
            voice_name, speed, noise_s, noise_w = speaker_params[current_role]
        else:
            voice_name, speed, noise_s, noise_w = speaker_params[None]

        logger.debug(f"Iniciando pipeline para voz '{voice_name}' com texto '{part[:50]}...'")

        # Etapa 1: Fonemização (CPU)
        t_fon = time.perf_counter()
        _, phoneme_ids, scales = await loop.run_in_executor(
            cpu_pool, phonemize_and_ids, voice_name, part, speed, noise_s, noise_w
        )
        t_fon_end = time.perf_counter()
        logger.debug(f"Fonemização levou {t_fon_end-t_fon:.4f}s, resultou em {len(phoneme_ids)} IDs")

        if not phoneme_ids:
            logger.warning(f"Segmento sem fonemas: '{part[:50]}...', devolvendo silêncio")
            return {'pcm_bytes': bytes(22050*2), 'sample_rate': 22050}

        # Etapa 2: Inferência (GPU)
        t_inf = time.perf_counter()
        sample_rate, pcm, synth_time, worker_id = await loop.run_in_executor(
            tts_pool, _synthesize_text_worker, voice_name, phoneme_ids, scales
        )
        logger.debug(f"Inferência GPU levou {synth_time:.4f}s (worker {worker_id})")

        with worker_metrics_lock:
            wm = worker_metrics[worker_id]
            wm["count"] += 1
            wm["total_synth_time"] += synth_time
            wm["jobs"].append(synth_time)

        with _active_lock:
            global _active_synthesis_count
            _active_synthesis_count += 1

        return {'pcm_bytes': pcm, 'sample_rate': sample_rate}

    results = await asyncio.gather(*[process_part(p) for p in parts])
    for seg in results:
        if seg: segments_data.append(seg)

    if not segments_data: raise HTTPException(400, "Nenhum segmento de áudio gerado.")

    ambient_dict = req.ambient.model_dump() if hasattr(req.ambient, 'model_dump') else req.ambient.dict()
    wav_bytes, mix_time = await loop.run_in_executor(mix_pool, mix_and_concat, segments_data, ambient_dict, SAMPLE_RATE_TARGET)

    t_total = time.perf_counter() - t_total_start
    with bench_lock:
        total_times.append(t_total)
        tts_wall_times.append(t_total - mix_time)
        mix_times.append(mix_time)
        queue_wait_times.append(0.0)

    logger.info(f"✅ Síntese concluída | total={t_total:.3f}s | mix={mix_time:.3f}s")
    return Response(content=wav_bytes, media_type="audio/wav")

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
    data = {
        "status": "ok",
        "gpu": True,
        "voices": list(voices_registry.keys()),
        "total": len(voices_registry)
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/bench")
async def bench():
    with bench_lock:
        ts = compute_stats(total_times)
        tts = compute_stats(tts_wall_times)
        ms = compute_stats(mix_times)
        qs = compute_stats(queue_wait_times)
    worker_data = dict(worker_metrics)
    data = {
        "benchmark_results": {
            "total": ts,
            "tts_wall": tts,
            "mix": ms,
            "queue_wait": qs
        },
        "gpu_workers": worker_data,
        "configuration": {
            "voices": list(voices_registry.keys()),
            "TTS_WORKERS": TTS_WORKERS,
            "CPU_WORKERS": CPU_WORKERS,
            "MIX_WORKERS": MIX_WORKERS
        }
    }
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(content=json_str, media_type="application/json")

@app.get("/stats")
async def get_stats():
    with stats_lock:
        if not stats:
            return Response(content=json.dumps({"message": "Nenhuma requisição ainda."}, indent=2), media_type="application/json")
        report = {}
        for key, values in stats.items():
            report[key] = compute_stats(values)
    return Response(content=json.dumps(report, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/logs")
async def get_logs():
    return Response(content=json.dumps({"logs": memory_handler.buffer}, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/gpu")
async def gpu_diagnostics():
    providers = ort.get_available_providers()
    nvidia_smi = ""
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        nvidia_smi = f"nvidia-smi não disponível: {e}"
    data = {
        "onnxruntime_version": ort.__version__,
        "providers": providers,
        "device": ort.get_device(),
        "nvidia_smi": nvidia_smi.strip(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "voices_loaded": list(voices_registry.keys())
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/workers")
async def workers():
    data = {
        "tts_workers": TTS_WORKERS,
        "cpu_workers": CPU_WORKERS,
        "mix_workers": MIX_WORKERS,
        "active_gpu_jobs": _active_synthesis_count,
        "per_worker": dict(worker_metrics)
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/resources")
async def resources():
    gpu_util = -1.0
    try:
        gpu_util = float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip())
    except Exception:
        pass
    data = {
        "gpu_utilization_percent": gpu_util,
        "cpu_cores_available": os.cpu_count(),
        "tts_workers": TTS_WORKERS,
        "cpu_workers": CPU_WORKERS,
        "mix_workers": MIX_WORKERS
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

@app.post("/reset_stats")
async def reset_stats():
    with bench_lock:
        total_times.clear()
        tts_wall_times.clear()
        mix_times.clear()
        queue_wait_times.clear()
    with stats_lock:
        stats.clear()
    with worker_metrics_lock:
        worker_metrics.clear()
    return Response(content=json.dumps({"message": "Estatísticas resetadas."}, indent=2), media_type="application/json")

@app.get("/carga")
async def carga():
    if not BENCH_DIR.exists():
        return Response(content=json.dumps({"message": "Nenhum teste de carga ainda."}, indent=2), media_type="application/json")
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return Response(content=json.dumps({"message": "Nenhum ficheiro de carga."}, indent=2), media_type="application/json")
    with open(files[0]) as f:
        data = json.load(f)
    result = {
        "file": files[0].name,
        "data": data,
        "available_files": [f.name for f in files]
    }
    return Response(content=json.dumps(result, indent=2, ensure_ascii=False), media_type="application/json")

@app.get("/carga_files")
async def carga_files():
    if BENCH_DIR.exists():
        files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    else:
        files = []
    return Response(content=json.dumps({"files": [f.name for f in files]}, indent=2), media_type="application/json")

@app.get("/carga/{file_name}")
async def carga_specific(file_name: str):
    p = BENCH_DIR / file_name
    if not p.exists():
        raise HTTPException(404, "Ficheiro não encontrado")
    with open(p) as f:
        data = json.load(f)
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

# ================= TESTE DE CARGA LOCAL =================
@app.post("/run_load_test")
async def run_load_test():
    import aiohttp, statistics, random
    RAMP_MAX, STEP, DUR, TO = 51, 5, 30, 30
    VOICE, AMB, AMB_VOL = "mulher_adulta", "ubs", -5.0
    DIALOGS = [
        "[paciente] Estou com dor de cabeça forte. Ele está assim há três dias, doutor.",
        "[paciente] Tenho tido muita tosse, [tosse]... [tosse]..  febre desde ontem.",
        "[paciente] Sinto falta de ar [inspiracao] ao caminhar. Ele já tem histórico de asma.",
        "[paciente] Estou muito cansada [inspiracao] e com falta de ar.  Ela parou de fumar há um mês.",
        "[paciente] A febre começou ontem à noite, depois que ele caiu.  Dói muito aqui!",
        "[paciente] O remédio não está fazendo efeito. Ele está tomando dipirona, mas não melhora.",
        "[paciente] Precisamos de uma receita para antibiótico.  Só dar o cu não está ajudando.",
        "[paciente] Meu peito dói quando respiro fundo [inspiracao]. E ele chupou um pau de 25 centímetros?",
        "[paciente] Quando posso voltar ao trabalho? Precisa de atestado por mais três dias.",
        "[paciente] Ele está com os exames alterados.  Vou precisar de cirurgia?"
    ]
    EFF = {"[tosse]":"tosse.wav","[suspiro]":"suspiro.wav","[inspiracao]":"inspiracao.wav"}
    BENCH_DIR.mkdir(exist_ok=True)
    fname = BENCH_DIR / f"carga_results_local_{int(time.time())}.json"
    results = []
    logger.info("Iniciando teste de carga local...")
    async with aiohttp.ClientSession() as sess:
        for concurrency in range(1, RAMP_MAX+1, STEP):
            sem = asyncio.Semaphore(concurrency)
            start = time.perf_counter()
            succ = 0
            fail = 0
            lats = []
            async def worker():
                nonlocal succ, fail, lats
                while time.perf_counter() - start < DUR:
                    async with sem:
                        d = random.choice(DIALOGS)
                        payload = {"voice":VOICE,"text":d,"effects":EFF,"ambient":{"enabled":True,"file":AMB,"volume_db":AMB_VOL}}
                        t0 = time.perf_counter()
                        try:
                            async with sess.post("http://localhost:8000/synthesize", json=payload, timeout=aiohttp.ClientTimeout(total=TO)) as resp:
                                if resp.status==200:
                                    succ += 1
                                    lats.append(time.perf_counter()-t0)
                                else:
                                    fail += 1
                        except Exception:
                            fail += 1
                        await asyncio.sleep(0)
            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.sleep(DUR)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            total = succ+fail
            avg = statistics.mean(lats) if lats else 0.0
            p95 = sorted(lats)[int(0.95*len(lats))] if lats else 0.0
            thr = succ/DUR
            err = fail/total if total else 1.0
            point = {
                "concurrency": concurrency,
                "throughput": thr,
                "avg_latency": avg,
                "p95_latency": p95,
                "error_rate": err,
                "total_requests": total,
                "success_count": succ
            }
            results.append(point)
            logger.info(f"  Throughput: {thr:.2f} req/s | Latência média: {avg:.3f}s | p95: {p95:.3f}s | Erros: {err*100:.1f}%")
            with open(fname,"w") as f:
                json.dump(results, f, indent=2)
    logger.info(f"Teste de carga local concluído. Resultados em {fname}")
    return Response(content=json.dumps({"message":"Teste concluído.","file":fname.name,"data":results}, indent=2), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
