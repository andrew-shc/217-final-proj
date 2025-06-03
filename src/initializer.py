import cv2
import numpy as np


class MapInitializer:
    def __init__(self):
        pass

    def initialize(self, kp1: np.ndarray, kp2: np.ndarray):
        """
        Assumes kp1[i] (Nx2) corresponds to kp2[i] (Nx2) for all i
        1. compute homography + fundamental matrix independently (RANSAC scheme)
        2. select model based off a heuristics
        3. retrieve motion hypothesis
        4. do various tests (i.e., chirality, reprojection error) to select final triangulated points or pass
        5. bundle adjustment to refine initial reconstruction
        """
        N = kp1.shape[0]

        H = self.compute_homography(kp1, kp2)
        print(H)

        ITERS = 1000
        for _ in range(ITERS):
            sample_indices = np.random.choice(N, 8, replace=False)
            p1_sample = kp1[:, sample_indices]
            p2_sample = kp2[:, sample_indices]
        



    def compute_homography(self, p1: np.ndarray, p2: np.ndarray):
        """
        via normalized DLT
        """
        p1 = self.__homogenize_points(p1)
        p1, T1 = self.__normalize_points(p1)
        p2 = self.__homogenize_points(p2)
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
        H = H / H[2,2]

        return H

        

    def compute_fundamental(self, p1: np.ndarray, p2: np.ndarray):
        """
        via 8-point algorithm
        """
        pass

    def full_bundle_adjustment(self):
        pass




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

    # Create a visualization image
    vis_image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, vis_image)

    return kp1_pts, kp2_pts

if __name__ == "__main__":
    kp1, kp2 = match_sift_keypoints_and_save_vis("final_project/217-final-proj/notes/parallax_1.jpg", "final_project/217-final-proj/notes/parallax_2.jpg", "output.jpg")
    # print(kp1[:10], kp2[:10])
    print(kp1.shape, kp2.shape)
    

    map = MapInitializer()
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

