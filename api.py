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
tts_wall_times: list[float] = []
mix_times: list[float] = []
queue_wait_times: list[float] = []
total_times: list[float] = []
bench_lock = threading.Lock()

worker_metrics = defaultdict(lambda: {"count": 0, "total_synth_time": 0.0, "jobs": []})
worker_metrics_lock = threading.Lock()

_active_synthesis_count = 0
_active_lock = threading.Lock()

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

# ====================== DETEÇÃO DE NÚCLEOS ======================
def _get_cores():
    """Retorna duas listas: physical_cores, ht_cores (apenas os lógicos)."""
    physical = set()
    all_logical = set()
    try:
        with open('/proc/cpuinfo') as f:
            current_physical = None
            for line in f:
                if line.startswith('processor'):
                    current_logical = int(line.split(':')[1].strip())
                    all_logical.add(current_logical)
                elif line.startswith('core id'):
                    current_physical = int(line.split(':')[1].strip())
                    if current_physical is not None:
                        physical.add(current_physical)
    except:
        pass

    if not physical:
        total = os.cpu_count()
        physical = set(range(total // 2))
        all_logical = set(range(total))

    # Os primeiros logicals de cada physical são os próprios físicos; os restantes são HT.
    physical_logical = set()
    ht_logical = set()
    for logical in sorted(all_logical):
        try:
            with open(f'/sys/devices/system/cpu/cpu{logical}/topology/thread_siblings_list') as f:
                siblings = list(map(int, f.read().strip().split(',')))
                if len(siblings) > 1 and logical != siblings[0]:
                    ht_logical.add(logical)
                else:
                    physical_logical.add(logical)
        except:
            # fallback: assume os primeiros metade são físicos
            if logical < len(all_logical)//2:
                physical_logical.add(logical)
            else:
                ht_logical.add(logical)
    return sorted(physical_logical), sorted(ht_logical)

_PHYSICAL, _HT = _get_cores()
logger.info(f"Núcleos físicos detetados: {_PHYSICAL}")
logger.info(f"Hyper-threads detetados: {_HT}")

# Afinidades a partir das variáveis de ambiente ou deteção automática
def _parse_cores(env_var, default_pool):
    val = os.getenv(env_var, "")
    if val:
        return [int(x.strip()) for x in val.split(",")]
    return default_pool

TTS_CORES = _parse_cores("TTS_CORES", _HT[:TTS_WORKERS])
CPU_CORES = _parse_cores("CPU_CORES", _HT[TTS_WORKERS:TTS_WORKERS+CPU_WORKERS])
MIX_CORES = _parse_cores("MIX_CORES", _PHYSICAL[:MIX_WORKERS])

logger.info(f"TTS_CORES (HT): {TTS_CORES}")
logger.info(f"CPU_CORES (HT): {CPU_CORES}")
logger.info(f"MIX_CORES (Físicos): {MIX_CORES}")

# ====================== POOL DE PROCESSOS (GPU) ======================
_manager = mp.Manager()
_tts_core_queue = _manager.Queue()
for core in TTS_CORES:
    _tts_core_queue.put(core)

def _init_tts_worker():
    core = _tts_core_queue.get()
    try:
        os.sched_setaffinity(0, {core})
    except Exception as e:
        logger.warning(f"Falha ao fixar worker TTS no núcleo {core}: {e}")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"

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

    import __main__
    __main__._worker_voice_cache = {}

def _synthesize_text_worker(voice_name, phoneme_ids, scales_tuple):
    """Executada no ProcessPoolExecutor: apenas inferência ONNX."""
    cache = getattr(sys.modules['__main__'], '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        setattr(sys.modules['__main__'], '_worker_voice_cache', cache)

    if voice_name not in cache:
        model_path, config_path = _get_voice_paths(voice_name)
        voice = PiperVoice.load(model_path, config_path, use_cuda=True)
        cache[voice_name] = voice
    voice = cache[voice_name]

    length_scale, noise_scale, noise_w = scales_tuple
    # Constroi os tensores diretamente
    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
    scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

    args = {
        "input": phoneme_ids_array,
        "input_lengths": phoneme_ids_lengths,
        "scales": scales,
    }
    if voice.config.num_speakers > 1:
        sid = np.array([0], dtype=np.int64)
        args["sid"] = sid

    t0 = time.perf_counter()
    audio = voice.session.run(None, args)[0].squeeze()
    audio = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
    synth_time = time.perf_counter() - t0

    worker_id = os.getpid()
    return voice.config.sample_rate, audio.tobytes(), synth_time, worker_id

# Cache de caminhos das vozes (preenchido no processo principal)
_VOICE_PATHS: Dict[str, Tuple[str, str]] = {}

def _get_voice_paths(voice_name):
    if voice_name not in _VOICE_PATHS:
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

# ---------- Registro de vozes (metadados) ----------
voices_registry: Dict[str, Dict] = {}

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

# ---------- Fonemização (CPU) ----------
def phonemize_and_ids(voice_name: str, text: str, speed: float,
                      noise_scale: float, noise_w_scale: float):
    """Executada no ThreadPoolExecutor da CPU: texto → IDs de fonemas."""
    model_path, config_path = _get_voice_paths(voice_name)
    # Podemos usar uma instância temporária apenas para fonemizar
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    voice = PiperVoice.load(model_path, config_path, use_cuda=False)  # CPU
    phonemes = voice.phonemize(text)
    phoneme_ids = voice.phonemes_to_ids(phonemes[0]) if phonemes else []
    # Não precisamos da sessão GPU aqui
    return voice_name, phoneme_ids, (speed, noise_scale, noise_w_scale)

# ---------- Mixagem com pydub (ThreadPool) ----------
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

# Pools globais
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
    loop = asyncio.get_running_loop()

    async def process_part(part: str) -> Optional[Dict]:
        nonlocal current_role
        part = part.strip()
        if not part:
            return None

        if part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if is_dialog and role in speaker_params:
                current_role = role
                return None
            if part in req.effects:
                effect_file = req.effects[part]
                vname = speaker_params[current_role][0] if is_dialog and current_role else req.voice
                return {'effect': effect_file, 'voice': vname}
            return None

        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto.")
            voice_name, speed, noise_s, noise_w = speaker_params[current_role]
        else:
            voice_name, speed, noise_s, noise_w = speaker_params[None]

        # Etapa 1: Fonemização (CPU)
        _, phoneme_ids, scales = await loop.run_in_executor(
            cpu_pool, phonemize_and_ids, voice_name, part, speed, noise_s, noise_w
        )

        # Etapa 2: Inferência (GPU)
        sample_rate, pcm, synth_time, worker_id = await loop.run_in_executor(
            tts_pool, _synthesize_text_worker, voice_name, phoneme_ids, scales
        )

        # Métricas do worker
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
        if seg:
            segments_data.append(seg)

    if not segments_data:
        raise HTTPException(400, "Nenhum segmento de áudio gerado.")

    ambient_dict = req.ambient.model_dump() if hasattr(req.ambient, 'model_dump') else req.ambient.dict()
    wav_bytes, mix_time = await loop.run_in_executor(mix_pool, mix_and_concat, segments_data, ambient_dict, SAMPLE_RATE_TARGET)

    t_total = time.perf_counter() - t_total_start
    with bench_lock:
        total_times.append(t_total)
        tts_wall_times.append(t_total - mix_time)  # aproximação
        mix_times.append(mix_time)
        queue_wait_times.append(0.0)

    record_global_stats(t_total, t_total - mix_time, mix_time, 0.0)
    logger.info(f"✅ Síntese concluída | total={t_total:.3f}s | mix={mix_time:.3f}s")
    return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoints de saúde (mantidos) ----------
@app.get("/started")
async def started(): return Response(status_code=200, content="started")
@app.get("/ready")
async def ready(): return Response(status_code=200 if voices_registry else 503, content="ready" if voices_registry else "loading model")
@app.get("/live")
async def live(): return Response(status_code=200, content="alive")
@app.get("/health")
async def health():
    return Response(content=json.dumps({"status":"ok","gpu":True,"voices":list(voices_registry.keys()),"total":len(voices_registry)}, indent=2), media_type="application/json")
@app.get("/bench")
async def bench():
    with bench_lock:
        ts = compute_stats(total_times)
        tts = compute_stats(tts_wall_times)
        ms = compute_stats(mix_times)
        qs = compute_stats(queue_wait_times)
    return Response(content=json.dumps({"benchmark_results":{"total":ts,"tts_wall":tts,"mix":ms,"queue_wait":qs},"gpu_workers":dict(worker_metrics),"configuration":{"voices":list(voices_registry.keys()),"TTS_WORKERS":TTS_WORKERS,"CPU_WORKERS":CPU_WORKERS,"MIX_WORKERS":MIX_WORKERS}}, indent=2), media_type="application/json")
@app.get("/stats")
async def get_stats():
    with stats_lock:
        if not stats: return Response(content=json.dumps({"message":"Nenhuma requisição ainda."}, indent=2), media_type="application/json")
        return Response(content=json.dumps({k:compute_stats(v) for k,v in stats.items()}, indent=2), media_type="application/json")
@app.get("/logs")
async def get_logs(): return Response(content=json.dumps({"logs":memory_handler.buffer}, indent=2), media_type="application/json")
@app.get("/gpu")
async def gpu():
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except: smi = "nvidia-smi não disponível"
    return Response(content=json.dumps({"onnxruntime_version":ort.__version__,"providers":ort.get_available_providers(),"device":ort.get_device(),"nvidia_smi":smi.strip(),"ld_library_path":os.environ.get("LD_LIBRARY_PATH",""),"voices_loaded":list(voices_registry.keys())}, indent=2), media_type="application/json")
@app.get("/workers")
async def workers():
    return Response(content=json.dumps({"tts_workers":TTS_WORKERS,"cpu_workers":CPU_WORKERS,"mix_workers":MIX_WORKERS,"active_gpu_jobs":_active_synthesis_count,"per_worker":dict(worker_metrics)}, indent=2), media_type="application/json")
@app.get("/resources")
async def resources():
    gpu_util = -1.0
    try: gpu_util = float(subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"], text=True).strip())
    except: pass
    return Response(content=json.dumps({"gpu_utilization_percent":gpu_util,"cpu_cores_available":os.cpu_count(),"tts_workers":TTS_WORKERS,"cpu_workers":CPU_WORKERS,"mix_workers":MIX_WORKERS}, indent=2), media_type="application/json")
@app.post("/reset_stats")
async def reset_stats():
    with bench_lock:
        total_times.clear(); tts_wall_times.clear(); mix_times.clear(); queue_wait_times.clear()
    with stats_lock: stats.clear()
    with worker_metrics_lock: worker_metrics.clear()
    return Response(content=json.dumps({"message":"Estatísticas resetadas."}, indent=2), media_type="application/json")
@app.get("/carga")
async def carga():
    if not BENCH_DIR.exists(): return Response(content=json.dumps({"message":"Nenhum teste de carga ainda."}, indent=2), media_type="application/json")
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files: return Response(content=json.dumps({"message":"Nenhum ficheiro de carga."}, indent=2), media_type="application/json")
    with open(files[0]) as f: data = json.load(f)
    return Response(content=json.dumps({"file":files[0].name,"data":data,"available_files":[f.name for f in files]}, indent=2), media_type="application/json")
@app.get("/carga_files")
async def carga_files():
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True) if BENCH_DIR.exists() else []
    return Response(content=json.dumps({"files":[f.name for f in files]}, indent=2), media_type="application/json")
@app.get("/carga/{file_name}")
async def carga_specific(file_name:str):
    p = BENCH_DIR / file_name
    if not p.exists(): raise HTTPException(404, "Ficheiro não encontrado")
    with open(p) as f: return Response(content=json.dumps(json.load(f), indent=2), media_type="application/json")

# ================= NOVO ENDPOINT: TESTE DE CARGA LOCAL =================
@app.post("/run_load_test")
async def run_load_test():
    import aiohttp, statistics, random
    RAMP_MAX = 51; STEP = 5; DUR = 30; TO = 30
    VOICE = "mulher_adulta"; AMB = "ubs"; AMB_VOL = -5.0
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
            sem = asyncio.Semaphore(concurrency); start = time.perf_counter(); succ = 0; fail = 0; lats = []
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
                                    succ += 1; lats.append(time.perf_counter()-t0)
                                else: fail += 1
                        except: fail += 1
                        await asyncio.sleep(0)
            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.sleep(DUR)
            for t in tasks: t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            total = succ+fail
            avg = statistics.mean(lats) if lats else 0.0
            p95 = sorted(lats)[int(0.95*len(lats))] if lats else 0.0
            thr = succ/DUR; err = fail/total if total else 1.0
            point = {"concurrency":concurrency,"throughput":thr,"avg_latency":avg,"p95_latency":p95,"error_rate":err,"total_requests":total,"success_count":succ}
            results.append(point)
            logger.info(f"  Throughput: {thr:.2f} req/s | Latência média: {avg:.3f}s | p95: {p95:.3f}s | Erros: {err*100:.1f}%")
            with open(fname,"w") as f: json.dump(results, f, indent=2)
    logger.info(f"Teste de carga local concluído. Resultados em {fname}")
    return Response(content=json.dumps({"message":"Teste concluído.","file":fname.name,"data":results}, indent=2), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
