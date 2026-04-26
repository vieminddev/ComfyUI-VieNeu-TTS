import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Voice list — populated dynamically after first model load.
# Fallback matches the voices shipped with VieNeu-TTS-v2-Turbo-GGUF.
# The SDK uses the full display name as the voice ID.
# ---------------------------------------------------------------------------
_VOICE_IDS: list = [
    "Bích Ngọc (Nữ - Miền Bắc)",
    "Phạm Tuyên (Nam - Miền Bắc)",
    "Thục Đoan (Nữ - Miền Nam)",
    "Xuân Vĩnh (Nam - Miền Nam)",
]

# ---------------------------------------------------------------------------
# Lazy singleton model loader
# ---------------------------------------------------------------------------
_tts_instance = None


def _get_tts():
    global _tts_instance, _VOICE_IDS
    if _tts_instance is None:
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise RuntimeError(
                "[VieNeu-TTS] vieneu package not found. "
                "Run: pip install vieneu"
            ) from exc
        print("[VieNeu-TTS] Loading model… (first run may take a moment)")
        _tts_instance = Vieneu()
        try:
            _VOICE_IDS = [vid for _, vid in _tts_instance.list_preset_voices()]
            print(f"[VieNeu-TTS] Loaded {len(_VOICE_IDS)} voices: {_VOICE_IDS}")
        except Exception as e:
            print(f"[VieNeu-TTS] Could not refresh voice list: {e}")
        print("[VieNeu-TTS] Model ready.")
    return _tts_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    """Convert a numpy audio array to ComfyUI AUDIO dict."""
    waveform = torch.from_numpy(audio_np).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)          # [C=1, T]
    elif waveform.dim() == 2:
        # soundfile returns [T, C] for multi-channel
        waveform = waveform.T.contiguous()        # [C, T]
    waveform = waveform.unsqueeze(0)              # [B=1, C, T]
    return {"waveform": waveform, "sample_rate": sample_rate}


def _comfy_to_wav_file(audio: dict, path: str) -> None:
    """Write a ComfyUI AUDIO dict to a WAV file (mono, first batch item)."""
    waveform = audio["waveform"]   # [B, C, T]
    sr = audio["sample_rate"]
    wav = waveform[0]              # [C, T]
    if wav.shape[0] > 1:           # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
    audio_np = wav.squeeze(0).numpy().astype(np.float32)
    sf.write(path, audio_np, sr, subtype="PCM_16")


def _apply_speed(audio: dict, speed: float) -> dict:
    """Time-scale audio by resampling (pitch-preserving via linear interpolation)."""
    if abs(speed - 1.0) < 1e-3:
        return audio
    waveform = audio["waveform"].float()   # [B, C, T]
    B, C, T = waveform.shape
    new_T = max(1, int(round(T / speed)))
    stretched = F.interpolate(waveform, size=new_T, mode="linear", align_corners=False)
    return {"waveform": stretched, "sample_rate": audio["sample_rate"]}


def _run_infer_to_comfy(tts, infer_kwargs: dict) -> dict:
    """Call tts.infer(**infer_kwargs), save to temp WAV, load back as ComfyUI AUDIO."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        audio_obj = tts.infer(**infer_kwargs)
        tts.save(audio_obj, tmp_path)
        audio_np, sr = sf.read(tmp_path, dtype="float32")
        return _numpy_to_comfy(audio_np, sr)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Node: VieNeuTTS  (preset voices)
# ---------------------------------------------------------------------------

class VieNeuTTSNode:
    """Generate speech from text using VieNeu-TTS preset voices."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        voices = _VOICE_IDS if _VOICE_IDS else ["(load model first)"]
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Xin chào, tôi là VieNeu TTS.",
                        "placeholder": "Nhập văn bản cần đọc…",
                    },
                ),
                "voice": (voices, {"default": voices[0]}),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
            }
        }

    def generate(self, text: str, voice: str, speed: float):
        if not text.strip():
            raise ValueError("[VieNeu-TTS] Text input is empty.")

        tts = _get_tts()
        voice_data = tts.get_preset_voice(voice)
        audio = _run_infer_to_comfy(tts, {"text": text, "voice": voice_data})
        audio = _apply_speed(audio, speed)
        return (audio,)


# ---------------------------------------------------------------------------
# Node: VieNeuTTSClone  (zero-shot voice cloning)
# ---------------------------------------------------------------------------

class VieNeuTTSCloneNode:
    """Clone a speaker's voice from a short reference audio clip."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone_voice"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Đây là giọng nói được clone từ audio mẫu.",
                        "placeholder": "Nhập văn bản cần đọc…",
                    },
                ),
                "ref_audio": ("AUDIO",),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Transcription of reference audio (optional)…",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
            }
        }

    def clone_voice(self, text: str, ref_audio: dict, ref_text: str, speed: float):
        if not text.strip():
            raise ValueError("[VieNeu-TTS] Text input is empty.")

        tts = _get_tts()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
        try:
            _comfy_to_wav_file(ref_audio, ref_path)

            infer_kwargs: dict = {"text": text, "ref_audio": ref_path}
            if ref_text.strip():
                infer_kwargs["ref_text"] = ref_text.strip()

            audio = _run_infer_to_comfy(tts, infer_kwargs)
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

        audio = _apply_speed(audio, speed)
        return (audio,)


# ---------------------------------------------------------------------------
# Node: VieNeuTTSUnloadModel  (memory management)
# ---------------------------------------------------------------------------

class VieNeuTTSUnloadModelNode:
    """Unload the VieNeu-TTS model from memory."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def unload(self):
        global _tts_instance
        if _tts_instance is not None:
            try:
                _tts_instance.close()
            except Exception:
                pass
            _tts_instance = None
            print("[VieNeu-TTS] Model unloaded.")
        return ()
