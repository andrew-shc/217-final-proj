import numpy as np

from util import Keyframe



class LocalMapping:
    def __init__(self, pts_3d: np.ndarray):
        self.covisibility_graph = CovisibilityGraph()
        self.pts_3d = pts_3d

    def insert_keyframe(self, kf: Keyframe):
        self.covisibility_graph.insert_keyframe(kf)

    def _map_points_culling(self):
        """
        of the provided points, it must satisfy:
        - more than 25% of the (predictied to be visible; covisible) frames
        - must be observed from at least three keyframes from map point creation
        """

        pass

    def _new_map_points(self):
        """
        new map points from triangulated ORB features by the connected keyframes via the covisibility graph
        """
        pass

    def _local_bundle_adjustment(self):
        """
        optimizes current keyframe + adjacent keyframe (via cov. graph) + all map points seen by these keyframes
        """
        pass

    def _local_keyframe_culling(self):
        """
        keyframes are discareded if 90% of its mapped points is visible/seen by other 3 keyframes in same/finer scale
        """
        pass

    def _triangulate(self, kf1: Keyframe, kf2: Keyframe):
        """
        Triangulate the set of points seen at location pts2L / pts2R in the
        corresponding pair of cameras. Return the 3D coordinates relative
        to this global coordinate system


        Parameters
        ----------
        pts2L : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (2,N) seen from camL camera

        pts2R : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (2,N) seen from camR camera

        camL : Camera
            The first "left" camera view

        camR : Camera
            The second "right" camera view

        Returns
        -------
        pts3 : 2D numpy.array (dtype=float)
            array containing 3D coordinates of the points in global coordinates

        """

        #
        #  your code goes here
        #
        #

        
        N = len(kf1.kp)
        pts3 = np.zeros((3, N))
        
        # image (pixel coords) -> image (intrinsic + homogeneous)
        qL = (kf1.p - kf1.c) / kf1.f
        qR = (kf2.p - kf2.c) / kf2.f
        qL = np.concatenate([qL, np.ones((1, qL.shape[1]))], axis=0)
        qR = np.concatenate([qR, np.ones((1, qR.shape[1]))], axis=0)

        for i in range(N):
            A = np.array([
                camL.R @ qL[:, i],
                -camR.R @ qR[:, i],
            ]).T
            b = camR.t - camL.t
            
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            zL, zR = Vh.T @ np.diag(1/S) @ U.T @ b

            # image (intrinsic + homogeneous) -> camera
            pL = zL * qL[:, i]
            pR = zR * qR[:, i]
            
            # camera -> world
            P1 = camL.R @ pL + camL.t.squeeze(1)
            P2 = camR.R @ pR + camR.t.squeeze(1)
            P = (P1+P2)/2
            
            pts3[:, i] = P
        
        return pts3
