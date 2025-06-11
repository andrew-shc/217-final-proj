import numpy as np
import cv2
import plotly.graph_objects as go



"""
This is the more complete OpenCV version of the initializer, which abstracts a lot of the
boring math, especially testing motion hypothesis.
"""


class MapInitializerCv:
    T_H = 5.99
    T_F = 3.84
    GAMMA = T_H

    def __init__(self, K: np.ndarray):
        self.K = K

    def initialize(
            self, kp1: np.ndarray, kp2: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assumes kp1[i] (Nx2) corresponds to kp2[i] (Nx2) for all i

        returns: R,t,M
        """
        H, H_inlier_mask = cv2.findHomography(kp1, kp2, method=cv2.RANSAC)
        F, F_inlier_mask = cv2.findFundamentalMat(kp1, kp2, method=cv2.FM_8POINT)
        p1, p2 = self.__homogenize_points(kp1), self.__homogenize_points(kp2)
        S_H = self.compute_H_score(p1, p2, H)
        S_F = self.compute_F_score(p1, p2, F)

        print("S_H:", S_H, "S_F:", S_F)

        # heuristic
        R_H = S_H/(S_H+S_F)
        print(R_H)
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
            
            R = Rs[best_idx]
            t = ts[best_idx]
            M = H
        else:
            # use fundamental

            E = self.K.T @ F @ self.K

            # Step 2: Recover relative pose (R, t)
            _, R, t, P_inlier_mask = cv2.recoverPose(E, kp1, kp2, self.K)

            M = F

        # given R,t  triangulate the points

        # >> insert triangulation code
        pts_3d = np.array([[0.0,0.0,0.0]])

        return (R, t, M, pts_3d)
        

    def compute_H_score(self, x1: np.ndarray, x2: np.ndarray, H: np.ndarray):
        H_inv = np.linalg.inv(H)

        # Project pts1 to image 2 and vice versa
        x1_to_2 = (H @ x1.T).T
        x2_to_1 = (H_inv @ x2.T).T

        # Normalize homogeneous coordinates
        x1_to_2 /= x1_to_2[:, 2:3]
        x2_to_1 /= x2_to_1[:, 2:3]

        # Compute squared distances
        d1 = np.sum((x2[:, :2] - x1_to_2[:, :2]) ** 2, axis=1)
        d2 = np.sum((x1[:, :2] - x2_to_1[:, :2]) ** 2, axis=1)

        score = self.rho(d1,T_M=self.T_H) + self.rho(d2,T_M=self.T_H)
        S_H = np.sum(score)

        return S_H
    
    def compute_F_score(self, p1: np.ndarray, p2: np.ndarray, F: np.ndarray):

        # Epipolar lines
        l2 = (F @ p1.T).T  # (N, 3) lines in image 2
        l1 = (F.T @ p2.T).T  # (N, 3) lines in image 1

        # Point-line distances squared
        d1 = np.sum(p2*l2, axis=1)**2 / (l2[:, 0]**2 + l2[:, 1]**2)
        d2 = np.sum(p1*l1, axis=1)**2 / (l1[:, 0]**2 + l1[:, 1]**2)

        score = self.rho(d1,T_M=self.T_F) + self.rho(d2,T_M=self.T_F)
        S_F = np.sum(score)

        return S_F

    def rho(self, d2: np.ndarray, T_M: float):
        """
        d2: symmetric transfer error
        """
        return np.where(d2 < T_M, self.GAMMA-d2, np.zeros_like(d2))

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

def stitch_images(img1, img2, H):
    """
    Stitches img1 onto img2 using the homography H (from img1 to img2).

    Args:
        img1: First image (to warp)
        img2: Second image (reference)
        H: 3x3 homography from img1 to img2

    Returns:
        stitched: New stitched image (img1 warped onto img2)
    """
    # Get image shapes
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get the corners of img1
    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    # Warp the corners to img2 space
    warped_corners = cv2.perspectiveTransform(corners_img1, H).squeeze(1)

    # Combine all corners to get bounding box
    all_corners = np.vstack((warped_corners, [[0, 0]], [[w2, 0]], [[w2, h2]], [[0, h2]])).reshape(-1, 2)
    [xmin, ymin] = np.floor(np.min(all_corners, axis=0)).astype(int)
    [xmax, ymax] = np.ceil(np.max(all_corners, axis=0)).astype(int)

    # Translation offset (if warped image goes negative)
    offset = [-xmin, -ymin]
    translation = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])

    # Warp img1
    output_size = (xmax - xmin, ymax - ymin)
    stitched = cv2.warpPerspective(img1, translation @ H, output_size)

    # Paste img2 into result
    stitched[offset[1]:offset[1]+h2, offset[0]:offset[0]+w2] = img2

    return stitched

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
    (R,t,M,pts_3d) = map.initialize(kp1, kp2)
    # assuming M is H

    img1 = cv2.imread('/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636750863555584.png')
    img2 = cv2.imread('/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636751663555584.png')

    # Assume H is known: from img1 to img2
    stitched = stitch_images(img1, img2, M)

    cv2.imwrite("H_stitched.png", stitched)


    x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
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


