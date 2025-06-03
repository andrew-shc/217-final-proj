import numpy as np


class MapInitializer:
    def __init__(self):
        pass

    def initialize(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        1. compute homography + fundamental matrix independently (RANSAC scheme)
        2. select model based off a heuristics
        3. retrieve motion hypothesis
        4. do various tests (i.e., chirality, reprojection error) to select final triangulated points or pass
        5. bundle adjustment to refine initial reconstruction
        """
        pass

    def compute_homography(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        via normalized DLT
        """
        pass

    def compute_fundamental(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        via 8-point algorithm
        """
        pass

    def full_bundle_adjustment(self):
        pass


if __name__ == "__main__":
    pass

