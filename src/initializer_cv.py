import numpy as np
import cv2

"""
This is the more complete OpenCV version of the initializer, which abstracts a lot of the
boring math, especially testing motion hypothesis.
"""


class MapInitializerCv:
    T_H = 5.99
    T_F = 3.84

    def __init__(self, K: np.ndarray):
        self.K = K

    def initialize(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        Assumes kp1[i] (Nx2) corresponds to kp2[i] (Nx2) for all i
        """
        H, H_inlier_mask = cv2.findHomography(kp1, kp2, method=cv2.RANSAC)
        F, F_inlier_mask = cv2.findFundamentalMat(kp1, kp2, method=cv2.FM_8POINT)
        p1, p2 = self.__homogenize_points(kp1), self.__homogenize_points(kp2)
        S_H = self.compute_H_score(p1, p2, H)
        S_F = self.compute_F_score(p1, p2, F)

        print("S_H:", S_H, "S_F:", S_F)

        # heuristic
        R_H = S_H/(S_H+S_F)
        if R_H > 0.45:
            # use homography

            # motion hypothesis from H (from pg.8): https://inria.hal.science/inria-00075698/PDF/RR-0856.pdf 
            # pain in an ass to implement it, hence we're skipping it

            # opencv uses a variation that only has at most 4 hypothesis
            _, Rs, ts, Ns = cv2.decomposeHomographyMat(H, self.K)


            best_idx = -1
            max_valid = -1

            for i in range(len(Rs)):
                R = Rs[i]
                t = ts[i].reshape(3, 1)

                P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = self.K @ np.hstack((R, t))

                pts4d_hom = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
                pts4d = pts4d_hom[:3] / pts4d_hom[3]

                depth1 = pts4d[2]
                depth2 = (R[2] @ pts4d + t[2])

                valid = (depth1 > 0) & (depth2 > 0)
                count = np.count_nonzero(valid)

                # print(1)
                if count > max_valid:
                    # print("===", count, kp1.shape[0])
                    max_valid = count
                    best_idx = i
            
            R_best = Rs[best_idx]
            t_best = ts[best_idx]

            return
        else:
            # use fundamental

            E = self.K.T @ F @ self.K

            # Step 2: Recover relative pose (R, t)
            p3d, R, t, P_inlier_mask = cv2.recoverPose(E, kp1, kp2, self.K)

            return
        

    def compute_H_score(self, p1: np.ndarray, p2: np.ndarray, H: np.ndarray):
        T_inv = np.linalg.inv(H)

        x2_proj = (H @ p1.T).T
        x1_proj = (T_inv @ p2.T).T

        x2_proj /= x2_proj[:, 2][:, None]
        x1_proj /= x1_proj[:, 2][:, None]

        err1 = np.sum((x1_proj[:, :2] - p1[:, :2]) ** 2, axis=1)
        err2 = np.sum((x2_proj[:, :2] - p2[:, :2]) ** 2, axis=1)
        rho = lambda d2: np.where(d2<self.T_H, self.T_H-d2, np.zeros_like(d2))

        S_H = np.sum(rho(err1)+rho(err2))

        return S_H
    
    def compute_F_score(self, p1: np.ndarray, p2: np.ndarray, F: np.ndarray):
        Fx1 = F @ p1.T
        Ftx2 = F.T @ p2.T
        x2tFx1 = np.sum(p2 * (F @ p1.T).T, axis=1)

        denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
        d = x2tFx1**2 / denom

        inliers = d < self.T_F
        S_F = np.sum(self.T_F - d[inliers])

        return S_F

    def __homogenize_points(self, p: np.ndarray):
        return np.stack([p[:,0],p[:,1],np.ones_like(p[:,0])], axis=1)

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
    # print(kp1.shape, kp2.shape)

    img1 = cv2.imread("/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636579763555584.png", cv2.IMREAD_GRAYSCALE)
    H, W = img1.shape
    # print(img1.shape)
    

    map = MapInitializerCv(
        # TODO: might want to find the actualy fx fy values
        K=np.array([
            [3024,  0.0, W/2],
            [ 0.0, 3024, H/2],
            [ 0.0,  0.0, 1.0],
        ])
    )
    map.initialize(kp1, kp2)
