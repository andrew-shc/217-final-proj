from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Keyframe:
    T: np.ndarray  # world to camera
    K: np.ndarray  # intrinsics
    # ORB keypoints + descriptors
    kp: list[cv2.KeyPoint]
    des: np.ndarray
    is_map_points: np.ndarray  # masking array
    map_points_idx: np.ndarray  # corresponding index to the map point


@dataclass
class MapPoints:  # batched (Nx...)
    X: np.ndarray  # 3d position (world/global)
    n: np.ndarray  # viewing direction
    des: np.ndarray  # corresponding ORB descriptor
    # ??? something about scale invariance limits of the ORB features
    d_max: np.ndarray
    d_min: np.ndarray

