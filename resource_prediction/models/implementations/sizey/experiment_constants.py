"""Experiment constants."""

# Code from jonathanbader
# Source: https://github.com/dos-group/sizey

from enum import Enum


class UnderPredictionStrategy(Enum):
    """Underprediction strategies."""

    DOUBLE = 1
    MAX_EVER_OBSERVED = 2


class OffsetStrategy(Enum):
    """Offset strategies."""

    STD_ALL = 1  # standard deviation
    MED_UNDER = 2  # median prediction error of underpredictions
    MED_ALL = 5  # median prediction error
    STD_UNDER = 6  # standard deviation of underpredictions
    DYNAMIC = 7  # tries all
