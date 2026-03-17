"""
Alpamayo-R1-10B inference module for OmniSight AV2 scene explanation.
"""

from .client import AlpamayoClient
from .inference import SceneInference

__all__ = ["AlpamayoClient", "SceneInference"]
