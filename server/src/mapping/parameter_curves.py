"""Parameter curve functions for audio-to-generation mapping."""

import math
from typing import Callable
from enum import Enum


class CurveType(str, Enum):
    """Available curve types for parameter mapping."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    THRESHOLD = "threshold"
    SMOOTHSTEP = "smoothstep"


def linear(value: float) -> float:
    """Linear mapping (identity)."""
    return value


def exponential(value: float, power: float = 2.0) -> float:
    """Exponential curve - more responsive at high values."""
    return math.pow(value, power)


def logarithmic(value: float, base: float = 10.0) -> float:
    """Logarithmic curve - more responsive at low values."""
    if value <= 0:
        return 0
    return math.log(1 + value * (base - 1)) / math.log(base)


def ease_in(value: float, power: float = 2.0) -> float:
    """Ease in (accelerating from zero)."""
    return math.pow(value, power)


def ease_out(value: float, power: float = 2.0) -> float:
    """Ease out (decelerating to target)."""
    return 1 - math.pow(1 - value, power)


def ease_in_out(value: float, power: float = 2.0) -> float:
    """Ease in and out (accelerate then decelerate)."""
    if value < 0.5:
        return math.pow(2, power - 1) * math.pow(value, power)
    return 1 - math.pow(-2 * value + 2, power) / 2


def threshold(value: float, thresh: float = 0.5) -> float:
    """Binary threshold - 0 below, 1 above."""
    return 1.0 if value >= thresh else 0.0


def smoothstep(value: float) -> float:
    """Smoothstep (Hermite interpolation)."""
    value = max(0, min(1, value))
    return value * value * (3 - 2 * value)


def get_curve_function(curve_type: CurveType, **kwargs) -> Callable[[float], float]:
    """Get curve function by type."""
    curves = {
        CurveType.LINEAR: lambda v: linear(v),
        CurveType.EXPONENTIAL: lambda v: exponential(v, kwargs.get("power", 2.0)),
        CurveType.LOGARITHMIC: lambda v: logarithmic(v, kwargs.get("base", 10.0)),
        CurveType.EASE_IN: lambda v: ease_in(v, kwargs.get("power", 2.0)),
        CurveType.EASE_OUT: lambda v: ease_out(v, kwargs.get("power", 2.0)),
        CurveType.EASE_IN_OUT: lambda v: ease_in_out(v, kwargs.get("power", 2.0)),
        CurveType.THRESHOLD: lambda v: threshold(v, kwargs.get("thresh", 0.5)),
        CurveType.SMOOTHSTEP: lambda v: smoothstep(v),
    }
    return curves.get(curve_type, linear)


def map_range(
    value: float,
    in_min: float = 0.0,
    in_max: float = 1.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    curve: CurveType = CurveType.LINEAR,
    **curve_kwargs,
) -> float:
    """Map a value from one range to another with optional curve."""
    # Normalize to 0-1
    normalized = (value - in_min) / (in_max - in_min) if in_max != in_min else 0
    normalized = max(0, min(1, normalized))

    # Apply curve
    curve_fn = get_curve_function(curve, **curve_kwargs)
    curved = curve_fn(normalized)

    # Map to output range
    return out_min + curved * (out_max - out_min)
