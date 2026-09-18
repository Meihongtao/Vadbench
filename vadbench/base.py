from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


class BaseVad(ABC):
    """Unified segment-level VAD interface."""

    name: str

    @abstractmethod
    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        """Detect speech segments.

        Args:
            audio: Wav path or mono float32 waveform in [-1, 1].
            sample_rate: Required when ``audio`` is an ndarray.

        Returns:
            List of ``SpeechSegment`` with times in seconds.
        """
