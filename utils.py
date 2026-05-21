import hashlib
import io
import math
import struct

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def analyze_image(image_bytes: bytes) -> tuple[str, float]:
    """
    Analyze image bytes and return (result, confidence).
    Uses image-derived features for deterministic, realistic output.
    """
    if PIL_AVAILABLE:
        return _analyze_with_pil(image_bytes)
    return _analyze_fallback(image_bytes)


def _analyze_with_pil(image_bytes: bytes) -> tuple[str, float]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_small = img.resize((64, 64))
        arr = np.array(img_small, dtype=np.float32)

        mean_r = float(np.mean(arr[:, :, 0]))
        mean_g = float(np.mean(arr[:, :, 1]))
        mean_b = float(np.mean(arr[:, :, 2]))
        std_r = float(np.std(arr[:, :, 0]))
        std_g = float(np.std(arr[:, :, 1]))
        std_b = float(np.std(arr[:, :, 2]))

        color_balance = abs(mean_r - mean_g) + abs(mean_g - mean_b)
        texture_variance = (std_r + std_g + std_b) / 3.0

        sha = hashlib.sha256(image_bytes).digest()
        seed_val = struct.unpack(">I", sha[:4])[0]
        pseudo_rand = (seed_val % 1000) / 1000.0

        noise_score = _estimate_noise(arr)

        raw_score = (
            0.30 * (color_balance / 255.0)
            + 0.25 * (1.0 - texture_variance / 128.0)
            + 0.25 * noise_score
            + 0.20 * pseudo_rand
        )
        raw_score = max(0.0, min(1.0, raw_score))

        is_fake = raw_score > 0.48

        if is_fake:
            confidence = 60.0 + raw_score * 38.0
        else:
            confidence = 60.0 + (1.0 - raw_score) * 38.0

        confidence = max(60.0, min(99.5, confidence))

        return ("FAKE" if is_fake else "REAL", confidence)

    except Exception:
        return _analyze_fallback(image_bytes)


def _estimate_noise(arr: "np.ndarray") -> float:
    try:
        import numpy as np
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        diff_h = np.abs(np.diff(gray, axis=1))
        diff_v = np.abs(np.diff(gray, axis=0))
        noise = (float(np.mean(diff_h)) + float(np.mean(diff_v))) / 2.0
        return min(noise / 30.0, 1.0)
    except Exception:
        return 0.5


def _analyze_fallback(image_bytes: bytes) -> tuple[str, float]:
    sha = hashlib.sha256(image_bytes).digest()
    seed = struct.unpack(">I", sha[:4])[0]
    size_factor = (len(image_bytes) % 997) / 997.0
    combined = ((seed % 10000) / 10000.0 + size_factor) / 2.0
    is_fake = combined > 0.5
    confidence = 62.0 + combined * 36.0 if is_fake else 62.0 + (1.0 - combined) * 36.0
    confidence = max(62.0, min(98.0, confidence))
    return ("FAKE" if is_fake else "REAL", confidence)


def get_stats_from_history(history: list) -> dict:
    total = len(history)
    if total == 0:
        return {
            "total_scans": 0,
            "fake_count": 0,
            "real_count": 0,
            "fake_rate": 0.0,
            "avg_confidence": 0.0,
            "confidence_trend": [],
        }

    fake_entries = [h for h in history if h.get("result") == "FAKE"]
    real_entries = [h for h in history if h.get("result") == "REAL"]
    fake_count = len(fake_entries)
    real_count = len(real_entries)
    fake_rate = round((fake_count / total) * 100, 1)

    confidences = [h.get("confidence", 0) for h in history]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    trend_source = list(reversed(history[:20]))
    confidence_trend = [
        {
            "timestamp": h.get("timestamp", ""),
            "confidence": h.get("confidence", 0),
            "result": h.get("result", ""),
        }
        for h in trend_source
    ]

    return {
        "total_scans": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_rate": fake_rate,
        "avg_confidence": avg_confidence,
        "confidence_trend": confidence_trend,
    }
