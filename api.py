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

# ========== FORÇA 1 THREAD NO ONNX RUNTIME (ANTES DE IMPORTAR) ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort

# Monkey patch do InferenceSession para forçar 1 thread
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

# ---------- Silenciar logs excessivos do ONNX Runtime ----------
ort.set_default_logger_severity(4)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# ---------- Logging da aplicação ----------
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
memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        memory_handler
    ]
)
logging.getLogger().handlers[0].setLevel(logging.INFO)
memory_handler.setLevel(logging.INFO)

logger = logging.getLogger("piper-api")
logger.setLevel(logging.DEBUG)

# ---------- Variáveis de ambiente ----------
MAX_GPU_JOBS = int(os.getenv("MAX_GPU_JOBS", "3"))
GPU_WORKERS = int(os.getenv("GPU_WORKERS", "2"))   # será usado para TTS
MIX_WORKERS = int(os.getenv("MIX_WORKERS", "4"))

POOL_INIT_SIZE = int(os.getenv("POOL_INIT_SIZE", "1"))
POOL_MAX_SIZE  = int(os.getenv("POOL_MAX_SIZE", str(GPU_WORKERS)))
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
tts_wall_times: list[float] = []
mix_times: list[float] = []
queue_wait_times: list[float] = []
total_times: list[float] = []
bench_lock = threading.Lock()

worker_metrics = defaultdict(lambda: {"count": 0, "total_synth_time": 0.0, "jobs": []})
worker_metrics_lock = threading.Lock()

_thread_local = threading.local()
_next_worker_id = 0
_worker_id_lock = threading.Lock()

active_synthesis_count = 0
active_synthesis_lock = threading.Lock()

def get_worker_id():
    if not hasattr(_thread_local, "worker_id"):
        with _worker_id_lock:
            global _next_worker_id
            _thread_local.worker_id = _next_worker_id
            _next_worker_id += 1
    return _thread_local.worker_id

def record_worker_job(synth_time: float):
    wid = get_worker_id()
    with worker_metrics_lock:
        wm = worker_metrics[wid]
        wm["count"] += 1
        wm["total_synth_time"] += synth_time
        wm["jobs"].append(synth_time)

def compute_worker_stats():
    with worker_metrics_lock:
        stats = {}
        for wid, data in worker_metrics.items():
            jobs = data["jobs"]
            if jobs:
                arr = np.array(jobs)
                stats[f"worker_{wid}"] = {
                    "requests_processed": data["count"],
                    "total_synth_time": data["total_synth_time"],
                    "avg_synth_time": float(np.mean(arr)),
                    "min_synth_time": float(np.min(arr)),
                    "max_synth_time": float(np.max(arr)),
                    "p95_synth_time": float(np.percentile(arr, 95)),
                }
            else:
                stats[f"worker_{wid}"] = {"requests_processed": 0}
        return stats

stats = defaultdict(list)
stats_lock = threading.Lock()

def record_global_stats(total: float, tts_wall: float, mix: float, queue_wait: float):
    with stats_lock:
        stats["total"].append(total)
        stats["tts_wall"].append(tts_wall)
        stats["mix"].append(mix)
        stats["queue_wait"].append(queue_wait)

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

# ====================== PROCESS POOL PARA TTS ======================
# Determinar núcleos físicos
def _get_physical_cores():
    physical = set()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('core id'):
                    physical.add(int(line.split(':')[1].strip()))
    except Exception:
        pass
    if not physical:
        # fallback: usa os primeiros N núcleos lógicos
        return list(range(os.cpu_count() // 2))
    return sorted(physical)

_PHYSICAL_CORES = _get_physical_cores()
_NUM_PHYSICAL = len(_PHYSICAL_CORES)

TTS_WORKERS = int(os.getenv("TTS_WORKERS", str(_NUM_PHYSICAL)))
MAX_GPU_JOBS = TTS_WORKERS   # o próprio pool já limita

# Fila de núcleos para os workers
_manager = mp.Manager()
_core_queue = _manager.Queue()
for _core in _PHYSICAL_CORES:
    _core_queue.put(_core)

# Cache de vozes local aos processos (será preenchido pelos workers)
def _init_tts_worker():
    # Configurar afinidade
    core = _core_queue.get()
    try:
        os.sched_setaffinity(0, {core})
    except Exception as e:
        logger.warning(f"Falha ao fixar worker no núcleo {core}: {e}")

    # Reforçar 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"

    # Aplicar monkey patch no onnxruntime dentro deste processo
    import onnxruntime as _ort
    _original_ort_session = _ort.InferenceSession
    def _patched_ort_session(model_path, sess_options=None, providers=None, **kwargs):
        if sess_options is None:
            sess_options = _ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return _original_ort_session(model_path, sess_options, providers=providers, **kwargs)
    _ort.InferenceSession = _patched_ort_session

    # Inicializa o cache de vozes local ao processo
    import __main__
    __main__._worker_voice_cache = {}

def _synthesize_text_worker(voice_name: str, text: str, speed: float,
                            noise_scale: float, noise_w_scale: float):
    """Função executada no ProcessPoolExecutor."""
    cache = getattr(sys.modules['__main__'], '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        setattr(sys.modules['__main__'], '_worker_voice_cache', cache)
    if voice_name not in cache:
        # Carregar voz – caminhos vêm do registo global (que está no processo principal)
        # Precisamos de ter acesso ao VOICE_PATHS. Vamos passar como argumento ou usar uma variável global do módulo.
        model_path, config_path = _get_voice_paths(voice_name)
        voice = PiperVoice.load(model_path, config_path, use_cuda=True)
        cache[voice_name] = voice
    voice = cache[voice_name]

    config = SynthesisConfig(
        length_scale=speed,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        volume=1.0
    )
    t0 = time.perf_counter()
    chunk_generator = voice.synthesize(text, syn_config=config)
    audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunk_generator)
    sample_rate = voice.config.sample_rate
    synth_time = time.perf_counter() - t0
    # Retornar também o PID para identificação do worker (opcional)
    worker_id = os.getpid()
    return sample_rate, audio_bytes, synth_time, worker_id

# Para que a função worker possa aceder aos caminhos das vozes, temos um dicionário global no módulo principal.
# No arranque, preenchemos _VOICE_PATHS com os caminhos e usamos uma função auxiliar que será chamada pelos workers.
_VOICE_PATHS: Dict[str, Tuple[str, str]] = {}

def _get_voice_paths(voice_name):
    if voice_name not in _VOICE_PATHS:
        # Carrega do disco (backup)
        voice_path = VOICES_DIR / voice_name
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
        _VOICE_PATHS[voice_name] = (model_path, config_path)
    return _VOICE_PATHS[voice_name]

# ---------- Registro de vozes no processo principal ----------
voices_registry: Dict[str, Dict] = {}   # apenas metadados

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

    _VOICE_PATHS[voice_name] = (model_path, config_path)
    return {"model_path": model_path, "config_path": config_path, "genero": genero, "path": voice_path}

def load_all_voices_metadata():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            name = item.name
            try:
                voices_registry[name] = load_voice_metadata(name, item)
                logger.debug(f"Voz registada: {name}")
            except Exception as e:
                logger.error(f"Falha ao registar voz {name}: {e}")

    for onnx_file in VOICES_DIR.glob("*.onnx"):
        name = onnx_file.stem
        if name in voices_registry:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            try:
                model_path = str(onnx_file)
                config_path = str(json_file)
                _VOICE_PATHS[name] = (model_path, config_path)
                voices_registry[name] = {"model_path": model_path, "config_path": config_path, "genero": "Personalizada", "path": VOICES_DIR}
                logger.debug(f"Voz raiz registada: {name}")
            except Exception as e:
                logger.error(f"Erro ao registar voz {name}: {e}")

    logger.info(f"Total de vozes registadas: {len(voices_registry)}")

# ---------- Mixagem com pydub (processo principal) ----------
def mix_and_concat(segments_data, ambient_cfg, target_rate=22050):
    t_start = time.perf_counter()
    combined = AudioSegment.empty()

    for data in segments_data:
        if 'pcm_bytes' in data:
            seg = AudioSegment(
                data=data['pcm_bytes'],
                sample_width=2,
                frame_rate=data['sample_rate'],
                channels=1
            )
            if seg.frame_rate != target_rate:
                seg = seg.set_frame_rate(target_rate)
            combined += seg
        elif 'effect' in data:
            voice_dir = voices_registry[data['voice']]["path"]
            effect_path = voice_dir / data['effect']
            if not effect_path.exists():
                effect_path = EFFECTS_DIR / data['effect']
            if not effect_path.exists():
                logger.debug(f"Efeito '{data['effect']}' não encontrado, ignorando.")
                continue
            try:
                effect_seg = AudioSegment.from_wav(str(effect_path))
                if effect_seg.frame_rate != target_rate:
                    effect_seg = effect_seg.set_frame_rate(target_rate)
                combined += effect_seg
            except Exception as e:
                logger.error(f"Erro ao carregar efeito '{data['effect']}': {e}")

    if len(combined) == 0:
        raise RuntimeError("Nenhum áudio foi gerado.")

    target_dbfs = -20.0
    if combined.dBFS != target_dbfs:
        combined = combined.apply_gain(target_dbfs - combined.dBFS)

    if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
        ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
        if ambient_path.exists():
            try:
                ambient = AudioSegment.from_wav(str(ambient_path))
                if ambient.frame_rate != target_rate:
                    ambient = ambient.set_frame_rate(target_rate)
                ambient = ambient + ambient_cfg.get('volume_db', -15.0)
                if len(ambient) < len(combined):
                    ambient = ambient * ((len(combined) // len(ambient)) + 1)
                ambient = ambient[:len(combined)]
                combined = combined.overlay(ambient)
            except Exception as e:
                logger.error(f"Erro ao aplicar ambiente: {e}")
        else:
            logger.debug(f"Ficheiro de ambiente '{ambient_cfg['file']}.wav' não encontrado.")

    with io.BytesIO() as buf:
        combined.export(buf, format="wav")
        wav_bytes = buf.getvalue()

    elapsed = time.perf_counter() - t_start
    return wav_bytes, elapsed

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
app = FastAPI(title="Piper TTS API (GPU)")

# Pools globais (inicializados no startup)
tts_pool: Optional[ProcessPoolExecutor] = None
mix_pool: Optional[ThreadPoolExecutor] = None
gpu_semaphore: asyncio.Semaphore = None   # não será usado com processos; mantemos por compatibilidade, mas pode ser removido

@app.on_event("startup")
async def startup():
    global tts_pool, mix_pool, gpu_semaphore
    load_all_voices_metadata()
    tts_pool = ProcessPoolExecutor(max_workers=TTS_WORKERS, initializer=_init_tts_worker)
    mix_pool = ThreadPoolExecutor(max_workers=MIX_WORKERS)
    gpu_semaphore = asyncio.Semaphore(MAX_GPU_JOBS)
    logger.info(f"Sistema pronto. TTS workers={TTS_WORKERS}, Mix={MIX_WORKERS}, MaxGPU={MAX_GPU_JOBS}")

# ================= ENDPOINT DE SÍNTESE =================
@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()
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
            ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_params[spk.role] = (spk.voice, spk.speed, ns, nw)
        for vn, _, _, _ in speaker_params.values():
            if vn not in voices_registry:
                raise HTTPException(404, f"Voz '{vn}' não encontrada")

    parts = re.split(r'(\[.*?\])', req.text)
    current_role = None
    segments_data = []
    tts_start = None
    tts_end = None
    total_queue_wait = 0.0
    loop = asyncio.get_running_loop()

    async def process_part(part: str) -> Tuple[Optional[Dict], float]:
        nonlocal current_role, tts_start, tts_end, total_queue_wait
        part = part.strip()
        if not part:
            return None, 0.0

        if part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if is_dialog and role in speaker_params:
                current_role = role
                return None, 0.0
            if part in req.effects:
                effect_file = req.effects[part]
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                return {'effect': effect_file, 'voice': vname}, 0.0
            return None, 0.0

        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto.")
            voice_name, speed, noise_s, noise_w = speaker_params[current_role]
        else:
            voice_name, speed, noise_s, noise_w = speaker_params[None]

        t0 = time.perf_counter()
        # Submeter ao pool de processos
        sample_rate, pcm, synth_time, worker_id = await loop.run_in_executor(
            tts_pool, _synthesize_text_worker, voice_name, part, speed, noise_s, noise_w
        )
        t1 = time.perf_counter()
        if tts_start is None:
            tts_start = t0
        tts_end = t1

        # Registar métrica do worker (worker_id)
        with worker_metrics_lock:
            wm = worker_metrics[worker_id]
            wm["count"] += 1
            wm["total_synth_time"] += synth_time
            wm["jobs"].append(synth_time)

        total_queue_wait += 0.0  # não temos pool.get() com processos, mas podemos medir o tempo de espera no pool
        return {'pcm_bytes': pcm, 'sample_rate': sample_rate}, t1 - t0

    tasks = [process_part(p) for p in parts]
    results = await asyncio.gather(*tasks)

    for seg, _ in results:
        if seg is not None:
            segments_data.append(seg)

    if not segments_data:
        raise HTTPException(400, "Nenhum segmento de áudio gerado.")

    tts_wall_time = (tts_end - tts_start) if (tts_start and tts_end) else 0.0

    ambient_dict = req.ambient.model_dump() if hasattr(req.ambient, 'model_dump') else req.ambient.dict()

    try:
        wav_bytes, mix_time = await loop.run_in_executor(
            mix_pool, mix_and_concat, segments_data, ambient_dict, SAMPLE_RATE_TARGET
        )
    except RuntimeError as e:
        logger.error(f"Erro na mixagem: {e}")
        raise HTTPException(500, "Falha na mixagem do áudio")

    t_total = time.perf_counter() - t_total_start

    with bench_lock:
        total_times.append(t_total)
        tts_wall_times.append(tts_wall_time)
        mix_times.append(mix_time)
        queue_wait_times.append(total_queue_wait)

    record_global_stats(t_total, tts_wall_time, mix_time, total_queue_wait)

    logger.info(
        f"✅ Síntese concluída | total={t_total:.3f}s | tts_wall={tts_wall_time:.3f}s | "
        f"mix={mix_time:.3f}s"
    )
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

# ---------- Benchmark ----------
@app.get("/bench")
async def bench():
    with bench_lock:
        total_stats = compute_stats(total_times)
        tts_stats = compute_stats(tts_wall_times)
        mix_stats = compute_stats(mix_times)
        queue_stats = compute_stats(queue_wait_times)

    worker_data = compute_worker_stats()

    data = {
        "benchmark_results": {
            "total": total_stats,
            "tts_wall": tts_stats,
            "mix": mix_stats,
            "queue_wait": queue_stats,
        },
        "gpu_workers": worker_data,
        "configuration": {
            "status": "ok",
            "voices": list(voices_registry.keys()),
            "workers": {
                "TTS_WORKERS": TTS_WORKERS,
                "MIX_WORKERS": MIX_WORKERS,
                "POOL_INIT_SIZE": POOL_INIT_SIZE,
                "POOL_MAX_SIZE": POOL_MAX_SIZE,
                "POOL_EXTRA_TTL_MINUTES": POOL_EXTRA_TTL_MINUTES
            },
            "gpu": True,
            "precision": "fp32",
            "sample_rate": SAMPLE_RATE_TARGET
        }
    }
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(content=json_str, media_type="application/json")

# ---------- Stats ----------
@app.get("/stats")
async def get_stats():
    with stats_lock:
        if not stats:
            return Response(content=json.dumps({"message": "Nenhuma requisição ainda."}, indent=2), media_type="application/json")
        report = {}
        for key, values in stats.items():
            report[key] = compute_stats(values)
    return Response(content=json.dumps(report, indent=2, ensure_ascii=False), media_type="application/json")

# ---------- Logs ----------
@app.get("/logs")
async def get_logs():
    return Response(content=json.dumps({"logs": memory_handler.buffer}, indent=2, ensure_ascii=False), media_type="application/json")

# ---------- GPU ----------
@app.get("/gpu")
async def gpu_diagnostics():
    providers = ort.get_available_providers()
    nvidia_smi = ""
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        nvidia_smi = str(e)
    data = {
        "onnxruntime_version": ort.__version__,
        "providers": providers,
        "device": ort.get_device(),
        "nvidia_smi": nvidia_smi.strip(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "voices_loaded": list(voices_registry.keys())
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

# ---------- Workers ----------
@app.get("/workers")
async def get_workers():
    data = {
        "tts_workers": TTS_WORKERS,
        "mix_workers": MIX_WORKERS,
        "active_gpu_jobs": active_synthesis_count,  # não atualizado com processos, mas mantemos
        "per_worker": compute_worker_stats()
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

# ---------- Recursos ----------
@app.get("/resources")
async def get_resources():
    gpu_util = -1.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        )
        gpu_util = float(out.strip())
    except Exception:
        pass

    cpu_util = -1.0
    try:
        import psutil
        cpu_util = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    data = {
        "gpu_utilization_percent": gpu_util,
        "cpu_utilization_percent": cpu_util,
        "cpu_cores_available": os.cpu_count(),
        "tts_workers": TTS_WORKERS,
        "mix_workers": MIX_WORKERS
    }
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")

# ---------- Reset stats ----------
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

# ---------- Teste de carga ----------
@app.get("/carga")
async def get_carga():
    if not BENCH_DIR.exists():
        return Response(content=json.dumps({"message": "Nenhum teste de carga foi executado ainda."}, indent=2), media_type="application/json")

    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return Response(content=json.dumps({"message": "Nenhum ficheiro de carga encontrado."}, indent=2), media_type="application/json")

    latest_file = files[0]
    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
        result = {
            "file": latest_file.name,
            "data": data,
            "available_files": [f.name for f in files]
        }
        return Response(content=json.dumps(result, indent=2, ensure_ascii=False), media_type="application/json")
    except Exception as e:
        logger.error(f"Erro ao ler ficheiro de carga {latest_file}: {e}")
        raise HTTPException(500, "Erro ao ler ficheiro de carga.")

@app.get("/carga_files")
async def list_carga_files():
    if not BENCH_DIR.exists():
        return Response(content=json.dumps({"files": []}, indent=2), media_type="application/json")
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return Response(content=json.dumps({"files": [f.name for f in files]}, indent=2), media_type="application/json")

@app.get("/carga/{file_name}")
async def get_specific_carga(file_name: str):
    file_path = BENCH_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, f"Ficheiro {file_name} não encontrado.")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return Response(content=json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json")
    except Exception as e:
        raise HTTPException(500, f"Erro ao ler ficheiro: {e}")

# ================= NOVO ENDPOINT: TESTE DE CARGA LOCAL =================
@app.post("/run_load_test")
async def run_load_test():
    """Dispara um teste de carga local e guarda os resultados na pasta bench."""
    import aiohttp
    import statistics
    import random

    RAMP_MAX_CONCURRENCY = 51
    RAMP_STEP = 5
    RAMP_STEP_DURATION = 30
    REQUEST_TIMEOUT = 30
    TEST_URL = "http://localhost:8000/synthesize"

    PACIENTE_VOICE = "mulher_adulta"
    AMBIENT_FILE = "ubs"
    AMBIENT_VOLUME_DB = -5.0

    DIALOGOS_BASE = [
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
    EFEITOS_DISPONIVEIS = {
        "[tosse]": "tosse.wav",
        "[suspiro]": "suspiro.wav",
        "[inspiracao]": "inspiracao.wav"
    }

    BENCH_DIR.mkdir(exist_ok=True)
    timestamp = int(time.time())
    carga_file = BENCH_DIR / f"carga_results_local_{timestamp}.json"
    results = []

    logger.info("Iniciando teste de carga local...")

    async with aiohttp.ClientSession() as session:
        for concurrency in range(1, RAMP_MAX_CONCURRENCY + 1, RAMP_STEP):
            logger.info(f"Testando com {concurrency} workers simultâneos...")
            sem = asyncio.Semaphore(concurrency)
            start_time = time.perf_counter()
            success = 0
            fail = 0
            latencies = []

            async def worker():
                nonlocal success, fail, latencies
                while time.perf_counter() - start_time < RAMP_STEP_DURATION:
                    async with sem:
                        dialogo = random.choice(DIALOGOS_BASE)
                        payload = {
                            "voice": PACIENTE_VOICE,
                            "text": dialogo,
                            "effects": EFEITOS_DISPONIVEIS,
                            "ambient": {"enabled": True, "file": AMBIENT_FILE, "volume_db": AMBIENT_VOLUME_DB}
                        }
                        t0 = time.perf_counter()
                        try:
                            async with session.post(TEST_URL, json=payload,
                                                   timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                                if resp.status == 200:
                                    success += 1
                                    latencies.append(time.perf_counter() - t0)
                                else:
                                    fail += 1
                        except Exception:
                            fail += 1
                        await asyncio.sleep(0)

            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.sleep(RAMP_STEP_DURATION)
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            total = success + fail
            if latencies:
                avg_lat = statistics.mean(latencies)
                p95 = sorted(latencies)[int(0.95 * len(latencies))]
            else:
                avg_lat = 0.0
                p95 = 0.0
            throughput = success / RAMP_STEP_DURATION
            error_rate = fail / total if total else 1.0

            point = {
                "concurrency": concurrency,
                "throughput": throughput,
                "avg_latency": avg_lat,
                "p95_latency": p95,
                "error_rate": error_rate,
                "total_requests": total,
                "success_count": success
            }
            results.append(point)
            logger.info(f"  Throughput: {throughput:.2f} req/s | Latência média: {avg_lat:.3f}s | p95: {p95:.3f}s | Erros: {error_rate*100:.1f}%")

            with open(carga_file, "w") as f:
                json.dump(results, f, indent=2)

    logger.info(f"Teste de carga local concluído. Resultados em {carga_file}")

    return Response(content=json.dumps({
        "message": "Teste de carga concluído.",
        "file": carga_file.name,
        "data": results
    }, indent=2, ensure_ascii=False), media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
