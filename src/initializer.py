import plotly.graph_objects as go
from typing import Callable
import cv2
import numpy as np


class MapInitializer:
    def __init__(self, K: np.ndarray):
        self.K = K

    def initialize(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        Assumes kp1[i] (Nx2) corresponds to kp2[i] (Nx2) for all i
        1. compute homography + fundamental matrix independently (RANSAC scheme)
        2. select model based off a heuristics
        3. retrieve motion hypothesis
        4. do various tests (i.e., chirality, reprojection error) to select final triangulated points or pass
        5. bundle adjustment to refine initial reconstruction
        """

        # H = self.compute_homography(kp1, kp2)
        # print(H)

        S_H, H = self.compute_model(kp1, kp2, model="homography")
        S_F, F = self.compute_model(kp1, kp2, model="fundamental")

        # heuristic
        R_H = S_H/(S_H+S_F)
        if R_H > 0.45:
            # use homography

            # retrieve 8 motion hypothesis

            # TODO
            print(H)
            # H, inlier_mask = cv2.findHomography(kp1, kp2, method=cv2.RANSAC)
            # inlier_mask = inlier_mask.ravel() == 1
            print(H)

            # dont even think about expanding this
            # tried this and it was way too tedious
            # just read this (from pg.8) and move on: https://inria.hal.science/inria-00075698/PDF/RR-0856.pdf 
            initial_hypothesis, Rs, ts, ns = cv2.decomposeHomographyMat(H, self.K)

            C1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # self.K @ 

            hypotheses = []
            for i in range(initial_hypothesis):
                R = Rs[i]
                t = ts[i].flatten()
                n = ns[i].flatten()
                # Add both signs for t
                hypotheses.append((R, +t, +n))
                hypotheses.append((R, -t, -n))

            best_in_front = -1
            best_P = None
            best_data = None
            for R, t, n in hypotheses:
                # Normalize pixel coordinates
                inp1 = kp1#[inlier_mask]
                inp2 = kp2#[inlier_mask]
                p1 = cv2.undistortPoints(inp1.reshape(-1, 1, 2), self.K, None).reshape(-1, 2).T
                p2 = cv2.undistortPoints(inp2.reshape(-1, 1, 2), self.K, None).reshape(-1, 2).T

                # Triangulate
                C2 = np.hstack((R, t[:, None]))  # self.K @ 
                P = cv2.triangulatePoints(C1, C2, p1, p2)  # shape (4, N)
                P = P[:3] / P[3]  # normalize homogeneous coordinates

                # cheirality check: positive depth in both cameras
                z1 = P[2, :]
                z2 = (R @ P + t[:, None])[2, :]
                in_front = np.sum((z1 > 0) & (z2 > 0))

                # test for parallax
                v1 = P  # v1: P - origin
                v2 = P - (-R.T @ t)[:, None]  # v2: P - center of Camera 2
                v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
                v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
                cos_theta = np.clip(np.sum(v1*v2, axis=1), -1.0, 1.0)
                angles = np.arccos(cos_theta)*180/np.pi
                parallax_ok = np.median(angles) > 20.0  # ORB-SLAM threshold deg = 1.0 (according to chatgpt)

                err1 = self.__reprojection_error(C1, P, p1.T)
                err2 = self.__reprojection_error(C2, P, p2.T)
                reproj_thresh = 4  # in pixels
                inliers = (err1 < reproj_thresh) & (err2 < reproj_thresh)
                reproj_err = (np.mean(err1[inliers]) + np.mean(err2[inliers])) / 2 if np.any(inliers) else np.inf
                
                # err1 = [self.__reprojection_error(C1, X, x) for X, x in zip(P.T, p1.T)]
                # err2 = [self.__reprojection_error(C2, X, x) for X, x in zip(P.T, p2.T)]
                # reproj_err = np.mean(err1) + np.mean(err2)
                # reproj_ok = reproj_err < 4  # reprojection threshold

                print(in_front, best_in_front, parallax_ok, np.median(angles), reproj_err, "baseline:", np.linalg.norm(t))
                if in_front >= best_in_front and parallax_ok:
                    best_in_front = in_front
                    best_data = (R, t, P)

            R,t,P = best_data
            print(P.shape)
            print(P[0].min(), P[0].max())
            print(P[1].min(), P[1].max())
            print(P[2].min(), P[2].max())
            # P = (P-np.mean(P, axis=1)[:, None])*10000.0
            # print(P[0].min(), P[0].max())
            # print(P[1].min(), P[1].max())
            # print(P[2].min(), P[2].max())

            x, y, z = P[0], P[1], P[2]
            cam1 = np.zeros(3)
            t /= np.linalg.norm(t)
            cam2 = (-R.T @ t).flatten()

            fig = go.Figure()

            # Plot 3D points
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2, color='blue'),
                name='Triangulated Points'
            ))

            fig.add_trace(go.Scatter3d(
                x=[cam1[0]], y=[cam1[1]], z=[cam1[2]],
                mode='markers+text',
                marker=dict(size=6, color='red'),
                text=['Camera 1'],
                textposition='top center',
                name='Camera 1'
            ))

            fig.add_trace(go.Scatter3d(
                x=[cam2[0]], y=[cam2[1]], z=[cam2[2]],
                mode='markers+text',
                marker=dict(size=6, color='green'),
                text=['Camera 2'],
                textposition='top center',
                name='Camera 2'
            ))

            # Baseline line
            fig.add_trace(go.Scatter3d(
                x=[cam1[0], cam2[0]],
                y=[cam1[1], cam2[1]],
                z=[cam1[2], cam2[2]],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Baseline'
            ))
            
            fig.update_layout(
                title='Triangulated Points (3xN Format)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )

            fig.show()

        else:
            # use fundamental

            # retrieve 4 motion hypothesis
            print("estimating F not implemented")
            pass

    def compute_model(self, kp1: np.ndarray, kp2: np.ndarray, model: str, ITERS = 1_000):
        """
        The RANSAC scheme part
        """
        N = kp1.shape[0]
        print(f"============ Computing model {model}")

        best_M_score = -1
        best_M = None

        for _ in range(ITERS):
            sample_indices = np.random.choice(N, 4 if model == "homography" else 8, replace=False)
            p1_sample = self.__homogenize_points(kp1[sample_indices])
            p2_sample = self.__homogenize_points(kp2[sample_indices])
            if model == "homography":
                M = self.compute_homography(p1_sample, p2_sample)
            elif model == "fundamental":
                M = self.compute_fundamental(p1_sample, p2_sample)
            else:
                raise TypeError(f"Invalid model to compute model: {model}")
        
            try:
                S_M = self.compute_score(p1_sample, p2_sample, M, model=model)
            except np.linalg.LinAlgError:
                continue  # error from attempting to invert singular matrix

            if S_M > best_M_score:
                print(S_M)
                best_M_score = S_M
                best_M = M
        
        return best_M_score, best_M

    def compute_score(self, p1: np.ndarray, p2: np.ndarray, M: np.ndarray, model: str):
        T_H = 5.99
        T_F = 3.84
        GAMMA = T_H  # invariant to model type (b/c they want the score to be homogenous for inlier region)
        if model == "homography":
            T_M = T_H


            T_inv = np.linalg.inv(M)

            x2_proj = (M @ p1.T).T
            x1_proj = (T_inv @ p2.T).T

            x2_proj /= x2_proj[:, 2][:, None]
            x1_proj /= x1_proj[:, 2][:, None]

            err1 = np.sum((x1_proj[:, :2] - p1[:, :2]) ** 2, axis=1)
            err2 = np.sum((x2_proj[:, :2] - p2[:, :2]) ** 2, axis=1)
            rho = lambda d2: np.where(d2<T_M, T_M-d2, np.zeros_like(d2))

            S_M = np.sum(rho(err1)+rho(err2))
        elif model == "fundamental":
            T_M = T_F

            Fx1 = M @ p1.T
            Ftx2 = M.T @ p2.T
            x2tFx1 = np.sum(p2 * (M @ p1.T).T, axis=1)

            denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
            d = x2tFx1**2 / denom

            inliers = d < T_M
            S_M = np.sum(T_M - d[inliers])
        else:
            raise TypeError(f"Invalid model to compute score: {model}")

        return S_M

    def compute_homography(self, p1: np.ndarray, p2: np.ndarray):
        """
        via normalized DLT + SVD
        """
        # p1 = self.__homogenize_points(p1)
        p1, T1 = self.__normalize_points(p1)
        # p2 = self.__homogenize_points(p2)
        p2, T2 = self.__normalize_points(p2)

        A = []
        for x1, x2 in zip(p1, p2):
            A.append([0,0,0,-x1[0],-x1[1],-1, x2[1]*x1[0], x2[1]*x1[1], x2[1]])
            A.append([ x1[0], x1[1], 1,0,0,0,-x2[0]*x1[0],-x2[0]*x1[1],-x2[0]])
        A = np.array(A)

        _U,_S,Vt = np.linalg.svd(A)
        h = Vt[-1]
        H_hat = h.reshape(3,3)
        H = np.linalg.inv(T2) @ H_hat @ T1
        return H / H[2,2]

    def compute_fundamental(self, p1: np.ndarray, p2: np.ndarray):
        """
        via 8-point algorithm + SVD
        """
        p1, T1 = self.__normalize_points(p1)
        p2, T2 = self.__normalize_points(p2)
        # p1 = np.linalg.inv(self.K) @ p1 @ self.K
        # p2 = np.linalg.inv(self.K) @ p2 @ self.K

        A = np.array([
            [x2[0]*x1[0], x2[0]*x1[1], x2[0],
             x2[1]*x1[0], x2[1]*x1[1], x2[1],
                   x1[0],       x1[1],    1 ]
            for x1, x2 in zip(p1, p2)  # note: switched the places
        ])

        # solve for F
        _, _, Vt = np.linalg.svd(A)
        F_hat = Vt[-1].reshape(3, 3)

        # enforce rank-2 constraint (b/c F is rank-2 by skew-symmetric t)
        U, S, Vt = np.linalg.svd(F_hat)
        S[-1] = 0
        F_hat = U @ np.diag(S) @ Vt

        F = T2.T @ F_hat @ T1
        return F / F[2,2]

    def full_bundle_adjustment(self):
        pass


    def __reprojection_error(self, C, X, x):
        N = X.shape[1]
        X_h = np.concatenate([X, np.ones((1, N))], axis=0)  # (4, N)
        x_proj = (C @ X_h).T  # (N, 3)
        x_proj = x_proj[:,:2] / x_proj[:,:2]
        return np.linalg.norm(x_proj[:,:2] - x, axis=1)

    def __homogenize_points(self, p: np.ndarray):
        return np.stack([p[:,0],p[:,1],np.ones_like(p[:,0])], axis=1)

    def __normalize_points(self, p: np.ndarray):
        """
        normalize points via T
        """
        mu = np.mean(p, axis=0)
        s = np.sqrt(2)/(np.sum(np.sqrt(np.sum((p-mu)**2, axis=1)), axis=0))
        T = np.array([
            [ s, 0, -s*mu[0]],
            [ 0, s, -s*mu[1]],
            [ 0, 0,  1],
        ])
        return (T @ p.T).T, T






def match_sift_keypoints_and_save_vis(
    image_path1,
    image_path2,
    save_path,
    ratio_thresh=0.75
):
    """
    Compute SIFT keypoints and descriptors from two images, match them using
    Lowe's ratio test, and return Nx2 arrays of matching keypoint positions.
    Also saves a visualization image showing matched keypoints.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        save_path (str): Path to save the output match visualization image.
        ratio_thresh (float): Lowe's ratio threshold. Default is 0.75.

    Returns:
        kp1_pts (np.ndarray): Array of shape (N, 2) of matched keypoints from image 1.
        kp2_pts (np.ndarray): Array of shape (N, 2) of matched keypoints from image 2.
    """
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    kp1_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    kp2_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # # Create a visualization image
    vis_image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("output.jpg", vis_image)
    vis_image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches[:0],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("output_2.jpg", vis_image)

    return kp1_pts, kp2_pts

if __name__ == "__main__":
    kp1, kp2 = match_sift_keypoints_and_save_vis(
        "/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636750863555584.png",
        "/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636751663555584.png",
        "output.jpg"
    )
    print(kp1.shape, kp2.shape)

    img1 = cv2.imread("/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636579763555584.png", cv2.IMREAD_GRAYSCALE)
    H, W = img1.shape
    print(img1.shape)
    

    map = MapInitializer(
        # TODO: might want to find the actualy fx fy values
        K=np.array([
            [3024,  0.0, W/2],
            [ 0.0, 3024, H/2],
            [ 0.0,  0.0, 1.0],
        ])
    )
    map.initialize(kp1, kp2)
    # map.compute_homography(kp1, kp2)
    # map.compute_homography(
    #     np.array([
    #         [10,10],
    #         [10,30],
    #         [30,10],
    #         [30,30],
    #     ]),
    #     np.array([
    #         [13,13],
    #         [13,33],
    #         [33,13],
    #         [33,33],
    #     ]),
    # )

