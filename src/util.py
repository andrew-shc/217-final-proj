from dataclasses import dataclass
import numpy as np


@dataclass
class Keyframe:
    T: np.ndarray  # world to camera
    K: np.ndarray  # intrinsics
    kp: np.ndarray  # ORB keypoints
    des: np.ndarray  # ORB descriptors


@dataclass
class MapPoints:  # batched (Nx...)
    X: np.ndarray  # 3d position (world/global)
    n: np.ndarray  # viewing direction
    des: np.ndarray  # corresponding ORB descriptor
    # ??? something about scale invariance limits of the ORB features
    d_max: np.ndarray
    d_min: np.ndarray

