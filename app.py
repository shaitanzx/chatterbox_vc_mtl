import os
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)
model_cache_path = config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True)

from pathlib import Path
import gradio as gr
import torch
import numpy as np
import tempfile
import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import shutil
import uuid
import librosa
import unicodedata
import re
from datetime import datetime
import soundfile as sf
import datetime
import engine  # TTS Engine interface
from models import CustomTTSRequest  # Pydantic models
import utils  # Utility functions

from ruaccent import RUAccent

# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(log_file_path_obj), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
current_config = {}
currentUiState = {}
appPresets = []
initialReferenceFiles = []
initialPredefinedVoices = []
hideChunkWarning = False
hideGenerationWarning = False
currentVoiceMode = 'predefined'
isGenerating = False
wavesurfer = None
currentAudioBlobUrl = None

reference_playing_state = {"is_playing": False, "current_file": None}

# --- SUPPORTED LANGUAGES ---
SUPPORTED_LANGUAGES = [
    "en", "ru", "de", "fr", "es", "pt", "tr", "zh", "ja", "ko",
    "hi", "ar", "sv", "nl", "pl", "it", "fi", "no", "ms", "he",
    "el", "da", "sw"
]

LANGUAGE_LABELS = {
    'en': "English", 'ru': "Russian", 'de': "German", 'fr': "French", 'es': "Spanish",
    'pt': "Portuguese", 'tr': "Turkish", 'zh': "Chinese", 'ja': "Japanese", 'ko': "Korean",
    'hi': "Hindi", 'ar': "Arabic", 'sv': "Swedish", 'nl': "Dutch", 'pl': "Polish",
    'it': "Italian", 'fi': "Finnish", 'no': "Norwegian", 'ms': "Malay", 'he': "Hebrew",
    'el': "Greek", 'da': "Danish", 'sw': "Swahili"
}

DISPLAY_TO_CODE = {name: code for code, name in LANGUAGE_LABELS.items()}

def extract_language_code(display_text: str) -> str:
    if " (" in display_text and display_text.endswith(")"):
        lang_name = display_text.split(" (")[0]
    else:
        lang_name = display_text  # на случай, если скобок нет

    return DISPLAY_TO_CODE.get(lang_name, display_text)

# --- Accentuation Support ---
try:
    accent_model = RUAccent()
    accent_model.load()
except Exception as e:
    logger.error(f"Failed to initialize RUAccent: {e}")
    accent_model = None

def convert_plus_to_accent(text: str) -> str:
    """Convert ruaccent '+vowel' markers to vowel with combining acute"""
    replacements = {
        '+а': 'а́', '+А': 'А́', '+е': 'е́', '+Е': 'Е́',
        '+ё': 'ё́', '+Ё': 'Ё́', '+и': 'и́', '+И': 'И́',
        '+о': 'о́', '+О': 'О́', '+у': 'у́', '+У': 'У́',
        '+ы': 'ы́', '+Ы': 'Ы́', '+э': 'э́', '+Э': 'Э́',
        '+ю': 'ю́', '+Ю': 'Ю́', '+я': 'я́', '+Я': 'Я́',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def load_custom_accents() -> Dict[str, str]:
    """Load custom accent fixes from YAML and dict files (из server.py)"""
    yaml_fixes = {}
    path = Path("accent_fixes.yaml")
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, str):
                            yaml_fixes[k.strip()] = v.strip()
        except Exception as e:
            logger.error(f"Failed to load accent_fixes.yaml: {e}")
    
    logger.info(f"Loaded {len(yaml_fixes)} custom accent fixes")
    return yaml_fixes

CUSTOM_ACCENTS = load_custom_accents()

def apply_custom_fixes(text: str) -> str:
    """Apply custom accent fixes"""
    text = unicodedata.normalize("NFC", text)
    items = [(k, v) for k, v in CUSTOM_ACCENTS.items() 
             if isinstance(k, str) and isinstance(v, str)]
    items.sort(key=lambda kv: len(kv[0]), reverse=True)
    for wrong, correct in items:
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text

def on_accent_click(text: str):
    if accent_model is None:
        gr.Error("⚠️ RUAccent model not loaded")
        return text
    
    try:
        raw_text = accent_model.process_all(text)
        accented_text = convert_plus_to_accent(raw_text)
        accented_text = apply_custom_fixes(accented_text)
        gr.Info("✅ Stresses are placed!")
        return accented_text
    except Exception as e:
        logger.error(f"Error in accentuate_text_endpoint: {e}", exc_info=True)
        gr.Error(f"⚠️ Accentuation failed: {str(e)}")
        return text

def get_ui_initial_data() -> Dict[str, Any]:

    logger.info("Request for initial UI data")

    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        
        # Load presets
        loaded_presets = []
        ui_static_path = Path(__file__).parent
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "languages": SUPPORTED_LANGUAGES,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data: {e}", exc_info=True)
        return {"error": "Failed to load initial data"}




def upload_reference_audio_endpoint(files: List[gr.File]) -> Dict[str, Any]:
    # upload reference audio
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames = []
    errors = []
    
    for file_info in files:
        if not file_info:
            continue
            
        # Extract filename from Gradio file object
        filename = os.path.basename(file_info)
        safe_filename = utils.sanitize_filename(filename)
        destination_path = ref_path / safe_filename
        
        try:
            if destination_path.exists():
                logger.info(f"File '{safe_filename}' already exists.")
                uploaded_filenames.append(safe_filename)
                continue
            
            # Copy file
            shutil.copy2(file_info, destination_path)
            logger.info(f"Saved uploaded file to: {destination_path}")
            
            # Validate
            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 600
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                destination_path.unlink(missing_ok=True)
                errors.append({"filename": safe_filename, "error": validation_msg})
            else:
                uploaded_filenames.append(safe_filename)
                
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    
    all_files = utils.get_valid_reference_files()
    return {
        "message": f"Processed {len(files)} file(s)",
        "uploaded_files": uploaded_filenames,
        "all_reference_files": all_files,
        "errors": errors
    }

def custom_tts_endpoint(
    text: str,
    voice_mode: str,
    predefined_voice_id: Optional[str] = None,
    reference_audio_filename: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    speed_factor: float = 1.0,
    seed: int = 0,
    language: str = "en",
    split_text: bool = False,
    chunk_size: int = 120,
    output_format: str = "mp3",
    output_sample_rate: Optional[int] = None,
    audio_name: Optional[str] = None,
    #silence_trimming: bool = False,
    #internal_silence_fix: bool = False,
    #unvoiced_removal: bool = False
) -> Tuple[Optional[str], str]:  # (audio_file_path, status_message)
    """Original TTS generation function from server.py"""
    
    global isGenerating
    
    if isGenerating:
        return None, "Generation is already in progress."
    
    isGenerating = True
    start_time = time.time()
    
    try:
        if not engine.MODEL_LOADED:
            return None, "TTS engine model is not currently loaded or available."
        
        audio_prompt_path = None
        if voice_mode == "predefined":
            if not predefined_voice_id:
                return None,None,False
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential_path = voices_dir / predefined_voice_id
            if not potential_path.is_file():
                return None, f"Predefined voice file '{predefined_voice_id}' not found."
            audio_prompt_path = potential_path
            
        elif voice_mode == "custom":
            if not reference_audio_filename:
                return None, "Missing 'reference_audio_filename' for 'custom' voice mode."
            ref_dir = get_reference_audio_path(ensure_absolute=True)
            potential_path = ref_dir / reference_audio_filename
            if not potential_path.is_file():
                return None, f"Reference audio file '{reference_audio_filename}' not found."
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 600)
            is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
            if not is_valid:
                return None, f"Invalid reference audio: {msg}"
            audio_prompt_path = potential_path
        
        if split_text and len(text) > (chunk_size * 1.5):
            text_chunks = utils.chunk_text_by_sentences(text, chunk_size)
        else:
            text_chunks = [text]
        
        all_audio_segments_np = []
        engine_output_sample_rate = None
        
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_audio_tensor, chunk_sr = engine.synthesize(
                    text=chunk,
                    audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    language=language,
                )
                
                if chunk_audio_tensor is None or chunk_sr is None:
                    return None, f"TTS engine failed to synthesize audio for chunk {i+1}."
                
                if engine_output_sample_rate is None:
                    engine_output_sample_rate = chunk_sr
                
                processed_audio_np = chunk_audio_tensor.cpu().numpy().squeeze()
                all_audio_segments_np.append(processed_audio_np)
                
            except Exception as e:
                return None, f"Error processing audio chunk {i+1}: {str(e)}"
        
        if not all_audio_segments_np:
            return None, "Audio generation resulted in no output."
        
        final_audio_np = (
            np.concatenate(all_audio_segments_np)
            if len(all_audio_segments_np) > 1
            else all_audio_segments_np[0]
        )
        output_format_str = output_format if output_format else get_audio_output_format()
        if output_sample_rate is not None:
            final_output_sample_rate = output_sample_rate
        else:
            final_output_sample_rate = get_audio_sample_rate()
    
        encoded_audio_bytes = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_output_sample_rate,
            output_format=output_format_str,
            target_sample_rate=final_output_sample_rate,  
            )
        
        if encoded_audio_bytes is None:
            return None, "Failed to encode audio to requested format."
        
        outputs_dir = get_output_path(ensure_absolute=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = audio_name or f"tts_output_{timestamp_str}"
        file_name = utils.sanitize_filename(f"{suggested_filename_base}.{output_format_str}")
        file_path = outputs_dir / file_name
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)
        
        generation_time = time.time() - start_time
        
        return str(file_path), f"✅ Audio generated successfully in {generation_time:.2f}s"
       
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}"
    finally:
        isGenerating = False



def show_notification(message: str, type: str = "info") -> Dict[str, str]:
    """Аналог showNotification из script.js"""
    icon = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️"
    }.get(type, "ℹ️")
    
    return {
        "message": f"{icon} {message}",
        "type": type,
        "timestamp": time.strftime("%H:%M:%S")
    }

def toggleVoiceOptionsDisplay(voice_mode: str) -> Tuple[Dict, Dict]:
    return (
        gr.update(visible=(voice_mode == "predefined")),
        gr.update(visible=(voice_mode == "custom"))
    )

def populatePredefinedVoices() -> List[str]:
    voices = utils.get_predefined_voices()
    return [voice.get("filename", "") for voice in voices]

def populateReferenceFiles() -> List[str]:
    files = utils.get_valid_reference_files()
    return files

def applyPreset(preset_name: str, presets: List[Dict[str, Any]]) -> tuple:
    for preset in presets:
        if preset.get("name") == preset_name:
            params = preset.get("params", {})
            temperature = float(params.get("temperature", 0.7))
            exaggeration = float(params.get("exaggeration", 1.0))
            cfg_weight = float(params.get("cfg_weight", 7.0))
            speed_factor = float(params.get("speed_factor", 1.0))
            seed = int(params.get("seed", -1))
            
            return (temperature, exaggeration, cfg_weight, speed_factor, seed)
    
    return (0.7, 1.0, 7.0, 1.0, -1)

def postprocess(audio_file,silence_trimming,internal_silence_fix,unvoiced_removal,output_format,config_audio_output_sample_rate,speed_factor,audio_name):
        speed_factor = float (speed_factor)
        config_audio_output_sample_rate = int (config_audio_output_sample_rate)
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', audio_file)
        audio_data, engine_output_sample_rate = librosa.load(audio_file, sr=None)
        if silence_trimming:
            audio_data = utils.trim_lead_trail_silence(
                audio_data, engine_output_sample_rate
            )
        
        if internal_silence_fix:
            audio_datap = utils.fix_internal_silence(
                audio_data, engine_output_sample_rate
            )

        if unvoiced_removal:
            audio_data = utils.remove_long_unvoiced_segments(
                audio_data, engine_output_sample_rate
            )
        if speed_factor != 1.0:
            try:
                import torch
                final_audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
                
                final_audio_tensor, _ = utils.apply_speed_factor(
                    final_audio_tensor, 
                    engine_output_sample_rate, 
                    speed_factor
                )
                audio_data = final_audio_tensor.cpu().numpy()
            except Exception as e:
                logger.error(f"Failed to apply speed factor: {e}", exc_info=True)

        output_format_str = output_format if output_format else get_audio_output_format()
        if config_audio_output_sample_rate is not None:
            final_output_sample_rate = config_audio_output_sample_rate
        else:
            final_output_sample_rate = get_audio_sample_rate()
    
        encoded_audio_bytes = utils.encode_audio(
            audio_array=audio_data,
            sample_rate=engine_output_sample_rate,
            output_format=output_format_str,
            target_sample_rate=final_output_sample_rate,
            )
        
        if encoded_audio_bytes is None:
            return None, None, gr.update (visible=True)
        
        outputs_dir = get_output_path(ensure_absolute=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
    
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = audio_name or f"tts_output_post_{timestamp_str}"
        file_name = utils.sanitize_filename(f"{suggested_filename_base}_post.{output_format_str}")
        file_path = outputs_dir / file_name
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)

        return file_path
        
def on_generate_click(
    text: str,
    voice_mode: str,
    predefined_voice: str,
    reference_file: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    speed_factor: float,
    seed: int,
    language: str,
    split_text: bool,
    chunk_size: int,
    output_format: str,
    config_audio_output_sample_rate: int,
    audio_name: str,
    silence_trimming: bool,
    internal_silence_fix: bool,
    unvoiced_removal: bool
) -> Tuple[Optional[str], str, Dict[str, str]]:

    if not text or text.strip() == "":
        return None
    
    if voice_mode == "predefined" and predefined_voice == "none":
        return None
    
    if voice_mode == "custom" and reference_file == "none":
        return None
        
    language=extract_language_code(language)
    audio_file, message = custom_tts_endpoint(
        text=text,
        voice_mode=voice_mode,
        predefined_voice_id=predefined_voice if predefined_voice != "none" else None,
        reference_audio_filename=reference_file if reference_file != "none" else None,
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        speed_factor=speed_factor,
        seed=seed,
        language=language,
        split_text=split_text,
        chunk_size=chunk_size,
        output_format=output_format,
        output_sample_rate=config_audio_output_sample_rate,
        audio_name=audio_name,
        #silence_trimming=silence_trimming,
        #internal_silence_fix=internal_silence_fix,
        #unvoiced_removal=unvoiced_removal
    )
    gr.Info(message)
    if audio_file !=None:
        file_path = postprocess(audio_file,silence_trimming,internal_silence_fix,unvoiced_removal,output_format,config_audio_output_sample_rate,speed_factor,audio_name)
    else:
        file_path = audio_file


    return gr.update (value=audio_file, visible=True),gr.update (value=file_path, visible=True),gr.update (visible=True)

def on_text_input(text: str) -> str:
    """Обработчик ввода текста (аналог из script.js)"""
    return str(len(text))

def on_reference_upload(files: List[gr.File]):
    try:
        result =  upload_reference_audio_endpoint(files)
        all_files = result.get("all_reference_files", [])
        uploaded_files = result.get("uploaded_files", [])
        
        if uploaded_files:
            default_selection = uploaded_files[0] if uploaded_files else "none"
            updated_options = all_files
            
            return gr.update(choices=updated_options,value=default_selection),gr.update(choices=updated_options)
        else:
            return gr.update(choices=populateReferenceFiles()),gr.update(choices=populateReferenceFiles())
            
    except Exception as e:
        logger.error(f"Error in reference upload: {e}", exc_info=True)
        return populateReferenceFiles(), show_notification(f"❌ Upload failed: {str(e)}", "error")

def toggle_voice_audio(selected_file: str, voice_mode: str) -> Tuple[Optional[str], str, Dict, Dict]:
    global reference_playing_state
    if not selected_file:
        gr.Warning("⚠️ Please select a file")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    if voice_mode == "predefined":
        base_path = get_predefined_voices_path(ensure_absolute=True)
    else: 
        base_path = get_reference_audio_path(ensure_absolute=True)
    
    file_path = base_path / selected_file
    if not file_path.exists():
        gr.Error(f"❌ File not found: {selected_file}")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    
    current_key = f"{voice_mode}_{selected_file}"
    
    if reference_playing_state["is_playing"] and reference_playing_state["current_key"] == current_key:
        reference_playing_state = {"is_playing": False, "current_key": None}
        gr.Info(f"⏸️ Stopped: {selected_file}")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)

    reference_playing_state = {"is_playing": True, "current_key": current_key}
    gr.Info(f"🎵 Playing: {selected_file}")
    
    return (
        str(file_path), 
        "⏸️ Play/Stop", 
        gr.update(visible=True),  
        gr.update(value=str(file_path), autoplay=True)  
    )
def reset_playback_on_mode_change(voice_mode: str) -> Tuple[str, str, Dict]:

    global reference_playing_state
    reference_playing_state = {"is_playing": False, "current_key": None}
    return "▶️ Play/Stop", "▶️ Play/Stop", gr.update(visible=False)

def voice_conversion(input_audio_path, target_voice_audio_path, chunk_sec=60, overlap_sec=0.1, disable_watermark=True, pitch_shift=0):
    vc_model = engine.get_or_load_vc_model()
    model_sr = vc_model.sr

    wav, sr = sf.read(input_audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != model_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=model_sr)
        sr = model_sr

    total_sec = len(wav) / model_sr

    if total_sec <= chunk_sec:
        wav_out = vc_model.generate(
            input_audio_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_wav = wav_out.squeeze(0).numpy()
        return model_sr, out_wav

    # chunking logic for long files
    chunk_samples = int(chunk_sec * model_sr)
    overlap_samples = int(overlap_sec * model_sr)
    step_samples = chunk_samples - overlap_samples

    out_chunks = []
    for start in range(0, len(wav), step_samples):
        end = min(start + chunk_samples, len(wav))
        chunk = wav[start:end]
        temp_chunk_path = f"temp_vc_chunk_{start}_{end}.wav"
        sf.write(temp_chunk_path, chunk, model_sr)
        out_chunk = vc_model.generate(
            temp_chunk_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_chunk_np = out_chunk.squeeze(0).numpy()
        out_chunks.append(out_chunk_np)
        os.remove(temp_chunk_path)

    # Crossfade join as before...
    result = out_chunks[0]
    for i in range(1, len(out_chunks)):
        overlap = min(overlap_samples, len(out_chunks[i]), len(result))
        if overlap > 0:
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            result[-overlap:] = result[-overlap:] * fade_out + out_chunks[i][:overlap] * fade_in
            result = np.concatenate([result, out_chunks[i][overlap:]])
        else:
            result = np.concatenate([result, out_chunks[i]])
    return model_sr, result



def voice_change(current_config):
    with gr.Row():            
        with gr.Accordion("🗣 Target Voice", open=True):
            voice_mode_radio = gr.Radio(
                choices=["predefined", "custom"],
                value="predefined",
                label="Select Voice Mode"
                )
            with gr.Group(visible=True) as predefined_group:
                with gr.Row():
                    predefined_voice_select = gr.Dropdown(
                        choices=populatePredefinedVoices(),
                        label="Predefined Voices",
                        interactive=True
                        )
                with gr.Row():    
                    predefined_play_btn = gr.Button("▶️ Play/Stop")

            with gr.Group(visible=False) as clone_group:
                with gr.Row():
                    reference_file_select = gr.Dropdown(
                        choices=populateReferenceFiles(),
                        label="Custom Audio Files",
                        interactive=True
                        )
                with gr.Row(): 
                    reference_play_btn = gr.Button("▶️ Play/Stop")
                with gr.Row():
                    reference_upload_btn = gr.UploadButton("📁 Upload Custom Audio",
                        file_types=[".wav", ".mp3"],
                        file_count="multiple",
                        visible=True
                        )
            reference_audio_player = gr.Audio(
                visible=False,
                label="",
                interactive=False,
                show_label=False,
                elem_id="reference-audio-player",
                autoplay=False  
                )  
            reference_audio_trigger = gr.Audio(
                visible=False,
                elem_id="reference-audio-trigger"
                ) 
            predefined_play_btn.click(
                fn=lambda file: toggle_voice_audio(file, "predefined"),
                inputs=[predefined_voice_select],
                outputs=[
                        reference_audio_player,
                        predefined_play_btn,    
                        reference_audio_player, 
                        reference_audio_player   
                        ]
                )
            reference_play_btn.click(
                fn=lambda file: toggle_voice_audio(file, "custom"),
                inputs=[reference_file_select],
                outputs=[
                        reference_audio_player,  
                        reference_play_btn,      
                        reference_audio_player,  
                        reference_audio_player  
                        ]
                )
            voice_mode_radio.change(
                fn=toggleVoiceOptionsDisplay,
                inputs=[voice_mode_radio],
                outputs=[predefined_group, clone_group]
                )  
    return voice_mode_radio,predefined_voice_select,reference_file_select,reference_upload_btn    

def create_gradio_interface():
    initial_data = get_ui_initial_data()
    if isinstance(initial_data, dict):
        current_config = initial_data.get("config", {})
        appPresets = initial_data.get("presets", [])
        languages = initial_data.get("languages", ["en"])
    else:
        current_config = {}
        appPresets = []
        languages = ["en"]

    language_options = []
    for lang_code in languages:
        label = LANGUAGE_LABELS.get(lang_code, lang_code)
        language_options.append(f"{label} ({lang_code})")
    
    with gr.Blocks(title="Chatterbox Server",theme=gr.themes.Base()) as demo:
        demo.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'dark');window.location.search = params.toString();}}"
        )    
        gr.Markdown(f"# 🎤 {get_ui_title()}")
    # === VC TAB: Voice Conversion Tab ===
        with gr.Tab("🎤 Voice Conversion (VC)"):
            gr.Markdown("## Voice Conversion\nConvert one speaker's voice to sound like another speaker using a target voice audio.")
            with gr.Row():
                with gr.Column():
                    vc_input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio (to convert)")
                with gr.Column():
                    voice_mode_radio_vc,predefined_voice_select_vc,reference_file_select_vc, upload_button_vc = voice_change(current_config)
            vc_pitch_shift = gr.Number(value=0, label="Pitch", step=0.5, interactive=True)
            disable_watermark_checkbox = gr.Checkbox(label="Disable Perth Watermark", value=True, visible=False)
            vc_convert_btn = gr.Button("Run Voice Conversion")
            vc_output_files = gr.Files(label="Converted VC Audio File(s)",visible=False)
            vc_output_audio = gr.Audio(label="VC Output Preview", interactive=True,visible=False,show_download_button=True)

            def _vc_wrapper(input_audio_path, disable_watermark, pitch_shift,voice_mode_vc,predefined_voice_id,reference_audio_filename):
                audio_prompt_path = None
                if voice_mode_vc == "predefined":
                    voices_dir = get_predefined_voices_path(ensure_absolute=True)
                    potential_path = voices_dir / predefined_voice_id
                    target_voice_audio_path = potential_path
            
                elif voice_mode_vc == "custom":
                    ref_dir = get_reference_audio_path(ensure_absolute=True)
                    potential_path = ref_dir / reference_audio_filename
                    max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 600)
                    is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
                    target_voice_audio_path = potential_path

                sr, out_wav = voice_conversion(
                    input_audio_path,
                    target_voice_audio_path,
                    disable_watermark=disable_watermark,
                    pitch_shift=pitch_shift
                    )
                base = os.path.splitext(os.path.basename(input_audio_path))[0]
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
                out_path = f"outputs/{base}_vc_{timestamp}.wav"
                sf.write(out_path, out_wav, sr)
                return [out_path], out_path  # Files and preview

            vc_convert_btn.click(lambda: (gr.update(interactive=False)),outputs=[vc_convert_btn]) \
                .then(
                    fn=_vc_wrapper,
                    inputs=[vc_input_audio, disable_watermark_checkbox, vc_pitch_shift,voice_mode_radio_vc,predefined_voice_select_vc,reference_file_select_vc],
                    outputs=[vc_output_files, vc_output_audio]) \
                    .then (lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[vc_convert_btn,vc_output_audio])

        with gr.Tab("🎵 MTL Generation"):        
            with gr.Row():
                gr.Markdown("### Text to synthesize")
            with gr.Row():
                gr.Markdown("Enter the text you want to convert to speech. For audiobooks, you can paste long chapters.")
            with gr.Row():    
                text_area = gr.Textbox(
                label="",
                value=current_config.get("ui_state", {}).get("last_text", "\u041A\u043E\u0433\u0434\u0430\u0301-\u0442\u043E \u0433\u0435\u0440\
    \u043E\u0301\u0439 \u0431\u044B\u043B \u043E\u0301\u0444\u0438\u0441\u043D\u044B\
    \u043C \u0441\u043E\u0442\u0440\u0443\u0301\u0434\u043D\u0438\u043A\u043E\u043C\
    , \u043D\u0435 \u0448\u0438\u0301\u0431\u043A\u043E \u0440\u0432\u0430\u0301\u0432\
    \u0448\u0438\u043C\u0441\u044F \u0447\u0435\u0433\u043E\u0301-\u0442\u043E \u0434\
    \u043E\u0441\u0442\u0438\u0433\u0430\u0301\u0442\u044C \u0432 \u0436\u0438\u0301\
    \u0437\u043D\u0438."),
                placeholder="Enter text here...",
                lines=8,
                max_lines=15,
                show_copy_button=False,
                elem_id="text"
                )
            with gr.Row():        
                char_count = gr.Textbox(
                    label="Characters",
                    value="94",
                    interactive=False,
                    scale=1,
                    elem_id="char-count"
                    )   
                    
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech",elem_id="generate-btn")
                accent_btn = gr.Button("🇷🇺 Stress")
            with gr.Row():        
                # Уведомления (аналог popup-msg)
                notification_display = gr.JSON(
                label="Notifications",
                value={},
                visible=False
                )
            with gr.Group():
                with gr.Row():                
                    split_text_toggle = gr.Checkbox(
                        label="Split text into chunks",
                        value=True
                            )
                with gr.Row():            
                    chunk_size_slider = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=120,
                        step=10,
                        label="Chunk Size",
                        visible=True
                        )
                        
            voice_mode_radio,predefined_voice_select,reference_file_select, upload_button_mtl = voice_change(current_config)

            with gr.Row():
                with gr.Accordion("🎛 Generation Parameters", open=True):
                    with gr.Row():
                        with gr.Column():
                            temperature_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                value=get_gen_default_temperature(),
                                step=0.01,
                                label="Temperature"
                                )   
                            cfg_weight_slider = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=get_gen_default_cfg_weight(),
                                step=0.01,
                                label="CFG Weight"
                                )
                            exaggeration_slider = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=get_gen_default_exaggeration(),
                                step=0.01,
                                label="Exaggeration"
                                )
                            seed_input = gr.Number(
                                value=get_gen_default_seed(),
                                label="Generation Seed (0 or -1 for random)"
                                )
                                    
                        with gr.Column():
                            language_select = gr.Dropdown(
                                choices=language_options,
                                value=current_config.get("generation_defaults", {}).get("language", "English (en)"),
                                label="Language",
                                interactive=True
                                )
                            config_audio_output_format = gr.Dropdown(
                                choices=["wav", "mp3", "opus"],
                                value=current_config.get("audio_output", {}).get("format", "mp3"),
                                label="Audio Output Format",
                                interactive=True
                                )
                            config_audio_output_sample_rate = gr.Number(
                                label="Audio Sample Rate",
                                value=current_config.get("audio_output", {}).get("sample_rate", 24000),
                                precision=0,
                                interactive=True
                                )
                                    
            with gr.Row():                    
                with gr.Accordion("⚙️ Postprocessing Parameters", open=True):
                    with gr.Row():
                        speed_factor_slider = gr.Slider(
                            minimum=0.25,
                            maximum=4.0,
                            value=get_gen_default_speed_factor(),
                            step=0.05,
                            label="Speed Factor"
                            )
                    with gr.Row():
                        silence_trimming = gr.Checkbox(
                            label="Silence Trimming",
                            value=current_config.get("audio_processing", {}).get("enable_silence_trimming", "False"),
                            interactive=True
                            )
                        internal_silence_fix = gr.Checkbox(
                            label="Internal Silence Fix",
                            value=current_config.get("audio_processing", {}).get("enable_internal_silence_fix", "False"),
                            interactive=True
                            )
                        unvoiced_removal = gr.Checkbox(
                            label="Unvoiced Removal",
                            value=current_config.get("audio_processing", {}).get("enable_unvoiced_removal", "False"),
                            interactive=True
                            )

            with gr.Accordion("📚 Example Presets", open=False):
                with gr.Row():
                    if appPresets:
                        preset_buttons = []
                        for preset in appPresets:
                            btn = gr.Button(
                                preset.get("name", "Unnamed"),
                                size="sm",
                                variant="secondary"
                                )
                            btn.click(
                                fn=lambda p=preset: applyPreset(p.get("name", ""), appPresets),
                                inputs=[],
                                outputs=[temperature_slider, exaggeration_slider, 
                                        cfg_weight_slider, speed_factor_slider, seed_input]
                                )

            with gr.Row():                
                with gr.Accordion("📁 Audio File Name", open=False):
                    audio_name_input = gr.Textbox(
                        label="Custom Audio Name",
                        placeholder="Enter custom name (without extension)",
                        value=""
                        )
            with gr.Row():
                #audio_output = gr.Audio(
                #    label="Generated Audio",
                #    type="filepath",
                #    interactive=True,
                #    visible=False,
                #    show_download_button=True
                #    )
                audio_output = gr.Audio(label="Generated Audio", interactive=True,visible=False,show_download_button=True,type="filepath")
                    
            with gr.Row():
                post_output = gr.Audio(
                    label="Postprocessed Audio",
                    type="filepath",
                    interactive=True,
                    visible=False,
                    show_download_button=True
                    )
            with gr.Row():
                post_btn = gr.Button("🎵 PostProcessing",visible=False)
            with gr.Accordion("💡 Tips & Tricks", open=False):
                gr.Markdown("""
                - For **Audiobooks**, use **MP3** format, enable **Split text**, and set a chunk size of ~250-500.
                - Use **Predefined Voices** for consistent, high-quality output.
                - For **Voice Cloning**, upload clean reference audio (`.wav`/`.mp3`). Quality of reference is key.
                - Experiment with **Temperature** and other generation parameters to fine-tune output.
                """)

        generate_btn.click(lambda: (gr.update(interactive=False),gr.update(interactive=False)),outputs=[generate_btn,post_btn]) \
            .then(fn=on_generate_click,inputs=[
                text_area,
                voice_mode_radio,
                predefined_voice_select,
                reference_file_select,
                temperature_slider,
                exaggeration_slider,
                cfg_weight_slider,
                speed_factor_slider,
                seed_input,
                language_select,
                split_text_toggle,
                chunk_size_slider,
                config_audio_output_format,
                config_audio_output_sample_rate,
                audio_name_input,
                silence_trimming,
                internal_silence_fix,
                unvoiced_removal
                ],
                outputs=[audio_output,post_output,post_btn]) \
            .then(lambda: (gr.update(interactive=True),gr.update(interactive=True)),outputs=[generate_btn,post_btn])

        post_btn.click(lambda: (gr.update(interactive=False),gr.update(interactive=False)),outputs=[generate_btn,post_btn]) \
            .then(fn=postprocess,inputs=[audio_output,silence_trimming,internal_silence_fix,unvoiced_removal,config_audio_output_format,config_audio_output_sample_rate,speed_factor_slider,audio_name_input], 
                    outputs=[post_output]) \
            .then (lambda: (gr.update(interactive=True),gr.update(interactive=True)),outputs=[generate_btn,post_btn])

        accent_btn.click(
            fn=on_accent_click,
            inputs=[text_area],
            outputs=[text_area]
        )

        text_area.change(
            fn=on_text_input,
            inputs=[text_area],
            outputs=[char_count]
        )
        upload_button_vc.upload(
            fn=on_reference_upload,
            inputs=[upload_button_vc],
            outputs=[reference_file_select_vc,reference_file_select]
            )
        upload_button_mtl.upload(
            fn=on_reference_upload,
            inputs=[upload_button_mtl],
            outputs=[reference_file_select,reference_file_select_vc]
            )

    
    return demo


def main():

    logger.info("Initializing TTS Server...")
    
    if not engine.load_model():
        logger.critical("CRITICAL: TTS Model failed to load on startup.")
        return
    
    logger.info("TTS Model loaded successfully via engine.")

    demo = create_gradio_interface()
    
    demo.launch(share=True)

if __name__ == "__main__":
    main()