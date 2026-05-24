"""MiniCPM-V 4.6 + Ascend engine session: loads HF model once, decodes via
the C++ engine binary per request."""

from .session import MinicpmvSession, EOS_TOKEN_IDS, LAYER_TYPES, VOCAB_SIZE

__all__ = ["MinicpmvSession", "EOS_TOKEN_IDS", "LAYER_TYPES", "VOCAB_SIZE"]
