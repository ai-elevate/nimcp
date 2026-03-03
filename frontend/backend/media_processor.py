"""Multimodal feature composition — combines text, audio, and visual features into 1024-dim vector."""
import math


def compose_multimodal_features(
    text_features: list[float] | None = None,
    audio_features: list[float] | None = None,
    visual_features: list[float] | None = None,
) -> list[float]:
    """Compose multimodal features into a 1024-dim vector.

    Layout: [text:768 | audio:128 | visual:128]
    """
    result = [0.0] * 1024

    # Text features: first 768 dims
    if text_features:
        for i, v in enumerate(text_features[:768]):
            result[i] = v

    # Audio features: dims 768-895
    if audio_features:
        for i, v in enumerate(audio_features[:128]):
            result[768 + i] = v

    # Visual features: dims 896-1023
    if visual_features:
        for i, v in enumerate(visual_features[:128]):
            result[896 + i] = v

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in result))
    if norm > 0:
        result = [v / norm for v in result]

    return result
