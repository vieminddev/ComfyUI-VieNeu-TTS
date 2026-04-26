"""
ComfyUI-VieNeu-TTS
~~~~~~~~~~~~~~~~~~
Custom nodes for VieNeu-TTS — Vietnamese + English text-to-speech with voice cloning.

Model: https://huggingface.co/pnnbao-ump/VieNeu-TTS
"""

try:
    from .nodes import (
        VieNeuTTSNode,
        VieNeuTTSCloneNode,
        VieNeuTTSUnloadModelNode,
        VieNeuTTSTextNormNode,
    )

    NODE_CLASS_MAPPINGS = {
        "VieNeuTTS": VieNeuTTSNode,
        "VieNeuTTSClone": VieNeuTTSCloneNode,
        "VieNeuTTSUnloadModel": VieNeuTTSUnloadModelNode,
        "VieNeuTTSTextNorm": VieNeuTTSTextNormNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "VieNeuTTS": "VieNeu TTS",
        "VieNeuTTSClone": "VieNeu TTS (Voice Clone)",
        "VieNeuTTSUnloadModel": "VieNeu TTS – Unload Model",
        "VieNeuTTSTextNorm": "VieNeu TTS – Text Normalize",
    }

    print("[VieNeu-TTS] Nodes registered: VieNeuTTS, VieNeuTTSClone, VieNeuTTSTextNorm, VieNeuTTSUnloadModel")

except Exception as exc:  # pragma: no cover
    print(f"[VieNeu-TTS] Failed to register nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
