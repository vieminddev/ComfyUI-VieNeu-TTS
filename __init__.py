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
    )

    NODE_CLASS_MAPPINGS = {
        "VieNeuTTS": VieNeuTTSNode,
        "VieNeuTTSClone": VieNeuTTSCloneNode,
        "VieNeuTTSUnloadModel": VieNeuTTSUnloadModelNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "VieNeuTTS": "VieNeu TTS",
        "VieNeuTTSClone": "VieNeu TTS (Voice Clone)",
        "VieNeuTTSUnloadModel": "VieNeu TTS – Unload Model",
    }

    print("[VieNeu-TTS] Nodes registered: VieNeuTTS, VieNeuTTSClone, VieNeuTTSUnloadModel")

except Exception as exc:  # pragma: no cover
    print(f"[VieNeu-TTS] Failed to register nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
