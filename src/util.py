from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Keyframe:
    T: np.ndarray  # world to camera (where the world is defined by the first frame)
    K: np.ndarray  # intrinsics 3x3
    # ORB keypoints + descriptors
    kp: list[cv2.KeyPoint]
    des: np.ndarray
    is_map_points: np.ndarray  # masking array
    map_points_idx: np.ndarray  # corresponding index to the map point

    @property
    def c(self):
        return self.K[[0,1],[2,2]]
    
    @property
    def f(self):
        return self.K[[0,1],[0,1]]
    
    @property
    def p(self):
        return np.array([i.pt for i in self.kp], dtype=np.float32)


@dataclass
class MapPoints:  # batched (Nx...)
    X: np.ndarray  # 3d position (world/global)
    n: np.ndarray  # viewing direction
    des: np.ndarray  # corresponding ORB descriptor
    # ??? something about scale invariance limits of the ORB features
    d_max: np.ndarray
    d_min: np.ndarray

class CovisibilityGraph:
    def __init__(self):
        pass