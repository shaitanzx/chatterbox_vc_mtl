# File: engine.py
# Core TTS model loading and speech generation logic.
import os
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
import logging
import random
import numpy as np
import torch
import gc
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from typing import Optional, Tuple
from pathlib import Path
from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)



from pathlib import Path
import torch
from safetensors.torch import load_file as load_safetensors

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.mtl_tts import Conditionals, SUPPORTED_LANGUAGES # Need to import these too

from chatterbox.vc import ChatterboxVC

class PatchedChatterboxTTS(ChatterboxMultilingualTTS):
    """
    An inherited class that fixes the attention implementation issue by overriding
    the `from_local` class method.
    """
    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'PatchedChatterboxTTS':
        print("🚀 Using PatchedChatterboxTTS.from_local to load the model.")
        ckpt_dir = Path(ckpt_dir)

        # --- This is the original code from the library ---
        ve = VoiceEncoder()
        ve.load_state_dict(
            ###########torch.load(ckpt_dir / "ve.pt", weights_only=True)
            torch.load(ckpt_dir / "ve.pt", map_location=device, weights_only=True)
        )
        ve.to(device).eval()
        
        # --- OUR FIX IS APPLIED HERE ---
        # 1. Create the T3Config
        t3_config = T3Config.multilingual()
        
        # 2. Patch the config object directly
        # This part is slightly different because T3 doesn't take the config directly for attn
        # We'll go back to patching the LlamaConfig inside T3's init, but called from our override
        
        # Let's use the better approach of patching the T3's config logic
        # For simplicity, we directly recreate the T3 object with the fix logic
        from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
        from transformers import LlamaConfig, LlamaModel
        
        hp = T3Config.multilingual()
        cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        cfg._attn_implementation = "eager" # Our patch
        
        # We need to manually recreate T3 since we can't inject the patched cfg easily
        # A simpler way is to just call the original method and then fix the model...
        # Let's try a cleaner override. We will replicate the method entirely.

        # The T3 class init needs to be fixed. So we create our own T3.
        class PatchedT3(T3):
            def __init__(self, hp=None):
                super().__init__(hp)
                # Override the transformer model with a patched config
                cfg = self.cfg
                cfg._attn_implementation = "eager"
                self.tfmr = LlamaModel(cfg)

        t3 = PatchedT3(T3Config.multilingual()) # Use our patched T3
        
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            ######torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
            torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)


# --- Global Module Variables ---
multilingual_model: Optional[PatchedChatterboxTTS] = None
MULTILINGUAL_MODEL_LOADED: bool = False
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

vc_model: Optional[ChatterboxVC] = None
VC_MODEL_LOADED: bool = False


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def load_model() -> bool:
    """
    Loads the multilingual TTS model by default.
    """
    global chatterbox_model, MODEL_LOADED, model_device, multilingual_model, MULTILINGUAL_MODEL_LOADED
    global vc_model, VC_MODEL_LOADED  # ← ДОБАВЛЕНО: глобальные переменные для VC

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # ↓↓↓ ПОЛУЧАЕМ ПУТЬ К КЭШУ НАПРЯМУЮ ИЗ CONFIG_MANAGER ↓↓↓
        model_cache_path = config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True)
        logger.info(f"📁 Путь к кэшу моделей: {model_cache_path}")
        
        # Проверяем существование и размер кэша
        from pathlib import Path
        cache_path = Path(model_cache_path)
        if cache_path.exists():
            # Подсчет общего размера кэша
            total_size = 0
            file_count = 0
            for file_path in cache_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            size_gb = total_size / (1024**3)
            size_mb = total_size / (1024**2)
            
            logger.info(f"📦 Размер кэша: {size_gb:.2f} GB ({size_mb:.0f} MB)")
            logger.info(f"📄 Файлов в кэше: {file_count}")
            
            # Показываем модели в кэше
            model_dirs = [d for d in cache_path.iterdir() if d.is_dir() and d.name.startswith("models--")]
            logger.info(f"📚 Найдено моделей в кэше: {len(model_dirs)}")
            
            for model_dir in model_dirs:
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                
                # Размер конкретной модели
                model_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                model_size_mb = model_size / (1024**2)
                
                # Проверяем snapshots
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        snapshot = snapshots[0]  # берем первый snapshot
                        snapshot_files = list(snapshot.rglob("*.*"))
                        logger.info(f"  └─ {model_name}: {model_size_mb:.1f} MB, {len(snapshot_files)} файлов")
                        # Показываем основные файлы модели
                        for file_path in snapshot_files[:3]:  # первые 3 файла
                            if file_path.is_file():
                                file_mb = file_path.stat().st_size / (1024**2)
                                logger.info(f"     • {file_path.name}: {file_mb:.1f} MB")
                else:
                    logger.info(f"  └─ {model_name}: {model_size_mb:.1f} MB (скачивается...)")
        else:
            logger.info("📭 Кэш не существует, будет создан при загрузке моделей")
        # ↑↑↑ ПОЛУЧАЕМ ПУТЬ К КЭШУ НАПРЯМУЮ ИЗ CONFIG_MANAGER ↑↑↑

        # Determine the device
        device_setting = config_manager.get_string("tts_engine.device", "auto")
        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")
        model_device = resolved_device_str
        logger.info(f"🎯 Устройство для моделей: {model_device}")

        # Загружаем TTS модель
        logger.info("⬇️  Загрузка TTS модели...")
        multilingual_model = PatchedChatterboxTTS.from_pretrained(device=model_device)
        chatterbox_model = multilingual_model
        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True

        logger.info(f"✅ TTS модель загружена на устройство: {model_device}")
        logger.info("🌐 Мультиязычная модель теперь используется для ВСЕХ языков.")
        
        # ↓↓↓ ДОБАВЛЕНО: Загрузка модели Voice Conversion ↓↓↓
        try:
            # Пытаемся загрузить VC модель
            logger.info("⬇️  Загрузка Voice Conversion модели...")
            vc_model = ChatterboxVC.from_pretrained(device=model_device)
            VC_MODEL_LOADED = True
            logger.info(f"✅ Voice Conversion модель загружена на устройство: {model_device}")
            
            # ↓↓↓ ВЫВОД ИНФОРМАЦИИ О ЗАГРУЖЕННЫХ МОДЕЛЯХ ↓↓↓
            logger.info("=" * 60)
            logger.info("📊 СТАТУС ЗАГРУЗКИ МОДЕЛЕЙ:")
            logger.info(f"  • TTS модель: {'✅ ЗАГРУЖЕНА' if MODEL_LOADED else '❌ НЕ ЗАГРУЖЕНА'}")
            logger.info(f"  • VC модель: {'✅ ЗАГРУЖЕНА' if VC_MODEL_LOADED else '❌ НЕ ЗАГРУЖЕНА'}")
            logger.info(f"  • Устройство: {model_device}")
            logger.info(f"  • Путь кэша: {model_cache_path}")
            
            # Память GPU
            if model_device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0)
                total_memory = gpu_memory.total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                reserved_memory = torch.cuda.memory_reserved() / (1024**3)
                
                logger.info(f"  • GPU память: {allocated_memory:.1f}/{total_memory:.1f} GB использовано")
                logger.info(f"  • GPU зарезервировано: {reserved_memory:.1f} GB")
            
            logger.info("=" * 60)
            # ↑↑↑ ВЫВОД ИНФОРМАЦИИ О ЗАГРУЖЕННЫХ МОДЕЛЯХ ↑↑↑
            
        except Exception as vc_e:
            logger.warning(f"⚠️ Не удалось загрузить Voice Conversion модель: {vc_e}")
            logger.warning("Вкладка Voice Conversion будет недоступна.")
            vc_model = None
            VC_MODEL_LOADED = False
            
            # Выводим информацию даже если VC не загрузилась
            logger.info("=" * 60)
            logger.info("📊 СТАТУС ЗАГРУЗКИ МОДЕЛЕЙ:")
            logger.info(f"  • TTS модель: {'✅ ЗАГРУЖЕНА' if MODEL_LOADED else '❌ НЕ ЗАГРУЖЕНА'}")
            logger.info(f"  • VC модель: ❌ НЕ ЗАГРУЖЕНА (ошибка: {str(vc_e)[:100]}...)")
            logger.info(f"  • Устройство: {model_device}")
            logger.info(f"  • Путь кэша: {model_cache_path}")
            logger.info("=" * 60)
        # ↑↑↑ ДОБАВЛЕНО: Загрузка модели Voice Conversion ↑↑↑

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки мультиязычной модели: {e}", exc_info=True)
        multilingual_model = None
        chatterbox_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_LOADED = False
        return False


def get_model_info() -> dict:
    """
    Возвращает информацию о загруженных моделях.
    Может быть вызвана из server_gradio.py для отображения в интерфейсе.
    """
    try:
        model_cache_path = config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True)
        cache_path = Path(model_cache_path)
        
        info = {
            "status": {
                "tts_loaded": MODEL_LOADED,
                "vc_loaded": VC_MODEL_LOADED if 'VC_MODEL_LOADED' in globals() else False,
                "device": model_device,
                "cache_path": str(model_cache_path)
            },
            "cache_info": {
                "exists": cache_path.exists(),
                "total_size_mb": 0,
                "model_count": 0,
                "models": []
            }
        }
        
        if cache_path.exists():
            # Размер кэша
            total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            info["cache_info"]["total_size_mb"] = total_size / (1024**2)
            
            # Модели в кэше
            model_dirs = [d for d in cache_path.iterdir() if d.is_dir() and d.name.startswith("models--")]
            info["cache_info"]["model_count"] = len(model_dirs)
            
            for model_dir in model_dirs:
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                model_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                
                model_info = {
                    "name": model_name,
                    "size_mb": model_size / (1024**2),
                    "path": str(model_dir)
                }
                
                # Проверяем snapshots
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        snapshot = snapshots[0]
                        model_info["snapshot"] = str(snapshot)
                        model_info["files"] = [f.name for f in snapshot.iterdir() if f.is_file()][:5]
                
                info["cache_info"]["models"].append(model_info)
        
        return info
        
    except Exception as e:
        return {"error": str(e)}


def load_multilingual_model() -> bool:
    """
    Loads the multilingual TTS model, unloads the standard model,
    and sets the multilingual model as the default for all languages.
    """
    global multilingual_model, MULTILINGUAL_MODEL_LOADED, model_device
    global chatterbox_model, MODEL_LOADED

    if MULTILINGUAL_MODEL_LOADED:
        logger.info("Multilingual TTS model is already loaded and set as default.")
        return True

    if model_device is None:
        logger.error("Main model device not determined. Load main model first.")
        return False

    if chatterbox_model is not None:
        logger.info("Unloading the standard ChatterboxTTS model to free up memory...")
        chatterbox_model = None
        MODEL_LOADED = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Standard model unloaded and memory cleared.")

    try:
        logger.info(f"Loading multilingual model (PatchedChatterboxTTS) on {model_device}...")

        multilingual_model = PatchedChatterboxTTS.from_pretrained(device=model_device)

        chatterbox_model = multilingual_model

        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True

        logger.info(f"PatchedChatterboxTTS model loaded successfully on {model_device}.")
        logger.info("This model will now be used for ALL languages, including English.")
        return True

    except Exception as e:
        logger.error(f"Error loading multilingual model: {e}", exc_info=True)
        multilingual_model = None
        chatterbox_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_LOADED = False
        return False
    

def load_vc_model() -> bool:
    """
    Загружает модель Voice Conversion.
    """
    global vc_model, VC_MODEL_LOADED, model_device
    
    if VC_MODEL_LOADED:
        logger.info("Voice Conversion model is already loaded.")
        return True
    
    if model_device is None:
        logger.error("Main model device not determined. Load main model first.")
        return False
    
    try:
        logger.info(f"Loading Voice Conversion model on {model_device}...")
        
        # Загружаем модель VC
        vc_model = ChatterboxVC.from_pretrained(device=model_device)
        VC_MODEL_LOADED = True
        
        logger.info(f"Voice Conversion model loaded successfully on {model_device}.")
        return True
        
    except Exception as e:
        logger.error(f"Error loading Voice Conversion model: {e}", exc_info=True)
        vc_model = None
        VC_MODEL_LOADED = False
        return False


def get_or_load_vc_model() -> Optional[ChatterboxVC]:
    """
    Получает или загружает модель Voice Conversion.
    """
    global vc_model, VC_MODEL_LOADED
    
    if not VC_MODEL_LOADED:
        if not load_vc_model():
            return None
    
    return vc_model



    
def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the currently loaded TTS model.
    If the multilingual model is loaded, it handles all languages.
    """
    global chatterbox_model, multilingual_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None
    
    active_model = chatterbox_model

    try:
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info("Using default (potentially random) generation behavior as seed is 0.")

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        is_multilingual = isinstance(active_model, ChatterboxMultilingualTTS)
        
        if is_multilingual:
            logger.info(f"Synthesizing with multilingual model for language: {language}")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language_id=language,
            )
        else:
            logger.info("Synthesizing with standard English model.")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        return wav_tensor, active_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def get_supported_languages() -> list:
    """
    Returns a list of all supported languages ​​for the UI.
    """
    langs = SUPPORTED_LANGUAGES
    if isinstance(langs, dict):
        return list(langs.keys())
    elif isinstance(langs, (list, tuple, set)):
        return list(langs)
    else:
        try:
            return list(getattr(langs, "keys")())
        except Exception:
            return ["en"]
