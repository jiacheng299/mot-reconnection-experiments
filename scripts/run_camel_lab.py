"""Run TrackLab with the metadata types in the two vetted CAMELTrack checkpoints."""

from collections import defaultdict
from typing import Any

import numpy as np
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
import torch
from tracklab.main import main
from yacs.config import CfgNode


if __name__ == '__main__':
    # Keep weights-only deserialization enabled and admit only known metadata types.
    allowed = [CfgNode, np.dtype, (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               type(np.dtype('float64')), type(np.dtype('float32')), type(np.dtype('int64')),
               DictConfig, Metadata, ContainerMetadata, AnyNode, defaultdict, dict, Any]
    with torch.serialization.safe_globals(allowed):
        main()
