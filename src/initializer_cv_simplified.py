import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt



"""
This is the simplified version from the initializer_cv.py after listing out additional assumptions (notably, omit F).
"""


class SimplifiedMapInitializerCv:
    def __init__(self, K: np.ndarray):
        self.K = K

    def estimate_initial_pose(
            self, kp1: np.ndarray, kp2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the initial R,t pose of the 2nd frame.

        Returns R,t,H
        """
        N = kp1.shape[0]

        H, _H_inlier_mask = cv2.findHomography(kp1, kp2, method=cv2.RANSAC)
        p1, p2 = self.__homogenize_points(kp1), self.__homogenize_points(kp2)

        # use homography

        # motion hypothesis from H (from pg.8): https://inria.hal.science/inria-00075698/PDF/RR-0856.pdf 
        # pain in an ass to implement it, hence we're skipping it

        # opencv uses a variation that only has at most 4 hypothesis that can be easily checked by number of points ahead
        _, Rs, ts, _ = cv2.decomposeHomographyMat(H, self.K)

        best_idx = -1
        max_valid = -1

        for i in range(len(Rs)):
            R = Rs[i]
            t = ts[i].reshape(3, 1)

            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.K @ np.hstack((R, t))

            pts3d = self.triangulate_initial_point_cloud(kp1, kp2, R, t).T
            z1 = pts3d[2]
            print(R.T.shape, pts3d.shape, t.shape)
            z2 = (R.T @ (pts3d - t))[2]
            print(z1.shape, z2.shape)
            count = np.sum((z1 < 0) & (z2 < 0))

            # pts4d_hom = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
            # pts4d = pts4d_hom[:3] / pts4d_hom[3]

            # depth1 = pts4d[2]
            # depth2 = (R[2] @ pts4d + t[2])

            # valid = (depth1 > 0) & (depth2 > 0)
            # count = np.count_nonzero(valid)

            print(">>>", count, kp1.shape[0])
            if count > max_valid:
                print("===", count, kp1.shape[0])
                max_valid = count
                best_idx = i
        
        R = Rs[best_idx]
        t = ts[best_idx]

        print("Best:", R, t)

        return R,t,H

    def triangulate_initial_point_cloud(
            self, kp1: np.ndarray, kp2: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Estimates the initial point cloud via the given pose and keypoints.
        
        Return 3D points (Nx3).
        """

        pts2L = self.__homogenize_points(kp1)
        pts2R = self.__homogenize_points(kp2)
        RL, tL, RR, tR = np.eye(3,3), np.zeros((3,1)), R, t

        N = pts2L.shape[0]
        pts3 = np.zeros((N, 3))
        
        # image (pixel coords) -> image (intrinsic + homogeneous)
        # qL = (pts2L - camL.c) / camL.f
        # qR = (pts2R - camR.c) / camR.f
        # qL = np.concatenate([qL, np.ones((1, qL.shape[1]))], axis=0)
        # qR = np.concatenate([qR, np.ones((1, qR.shape[1]))], axis=0)
        
        inv_K = np.linalg.inv(self.K)

        qL = (inv_K @ pts2L.T)
        qR = (inv_K @ pts2R.T)

        for i in range(N):
            A = np.array([
                RL @ qL[:, i],
                -RR @ qR[:, i],
            ]).T
            b = tR - tL
            
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            zL, zR = Vh.T @ np.diag(1/S) @ U.T @ b

            # image (intrinsic + homogeneous) -> camera
            pL = zL * qL[:, i]
            pR = zR * qR[:, i]
            
            # camera -> world
            P1 = RL @ pL + tL.squeeze(1)
            P2 = RR @ pR + tR.squeeze(1)
            P = (P1+P2)/2
            
            pts3[i, :] = P
        
        return pts3


    def __homogenize_points(self, p: np.ndarray):
        return np.stack([p[:,0],p[:,1],np.ones_like(p[:,0])], axis=1)





def match_sift_keypoints_and_save_vis(
    img1,
    img2,
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

def visualize_point_cloud(pts_3d):
    pts_3d = pts_3d.T
    x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]

    fig = go.Figure()

    # Plot 3D points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Triangulated Points'
    ))
    
    fig.update_layout(
        title='Triangulated Points (3xN Format)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(title='X', range=[-1,1]),
            yaxis=dict(title='Y', range=[-1,1]),
            zaxis=dict(title='Z', range=[-6,3]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def generate_plane_square(
    p0,                # (3,) point on the plane (centre of the square)
    n,                 # (3,) plane normal (any length)
    side=2.0,          # side length of the square patch
    n_pts=1_000,       # number of random points to sample
    seed=None          # RNG seed for repeatability
):
    """
    Return an (n_pts, 3) point cloud lying on a square patch of a plane.

    The plane is (x – p0)·n = 0 and the square is centred at p0.
    """
    rng = np.random.default_rng(seed)

    # 1) Build two orthonormal in-plane axes u and v
    n = n / np.linalg.norm(n)
    helper = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)                       # already unit

    # 2) Uniformly sample inside the square
    half = side / 2.0
    uu = rng.uniform(-half, half, n_pts)
    vv = rng.uniform(-half, half, n_pts)

    pts = p0 + uu[:, None] * u + vv[:, None] * v  # (N, 3)

    return pts

def project(pts3: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    """
    Project the given 3D points in world coordinates into the specified camera    

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    Returns
    -------
    pts2 : 2D numpy.array (dtype=float)
        Image coordinates of N points stored in a array of shape (2,N)

    """

    assert(pts3.shape[0]==3)

    # get point location relative to camera
    pcam = R.T @ (pts3 - t)
        
    # project
    p = K @ pcam
    pts2 = p[0:2] / p[2,:]
    
    assert(pts2.shape[1]==pts3.shape[1])
    assert(pts2.shape[0]==2)

    return pts2

def roty(theta):
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
    return R

def main():
    ########################################################################################
    ###    Synthetic data generation
    ########################################################################################
    H, W = 1080, 1920
    K=np.array([
        [3024,  0.0, W/2],
        [ 0.0, 3024, H/2],
        [ 0.0,  0.0, 1.0],
    ])

    R=roty(20*np.pi/180)
    t=np.array([[2.0,-.1,-1.0]]).T

    # gt_pts_3d = generate_hemisphere(1, np.array([[0,0,-5]]).T, 1000)
    gt_pts_3d = generate_plane_square(np.array([0,0,-5]), np.array([0,0,1])).T
    synth_kp1 = project(gt_pts_3d, np.eye(3), np.zeros((3,1)), K).T
    synth_kp2 = project(gt_pts_3d, R, t, K).T
    viewport_mask = (
        (0 <= synth_kp1[:,0]) & (synth_kp1[:,0] <= W) & (0 <= synth_kp1[:,1]) & (synth_kp1[:,1] <= H) &
        (0 <= synth_kp2[:,0]) & (synth_kp2[:,0] <= W) & (0 <= synth_kp2[:,1]) & (synth_kp2[:,1] <= H)
    )
    synth_kp1 = synth_kp1[viewport_mask]
    synth_kp2 = synth_kp2[viewport_mask]
    synth_pts_3d = (gt_pts_3d.T)[viewport_mask]


    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].scatter(synth_kp1[:,0],synth_kp1[:,1], s=1)
    axs[1].scatter(synth_kp2[:,0],synth_kp2[:,1], s=1)
    axs[0].set_xlim(0, W)
    axs[0].set_ylim(0, H)
    axs[0].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(0, W)
    axs[1].set_ylim(0, H)
    axs[1].set_aspect('equal', adjustable='box')
    axs[0].set_title('1st')
    # axs[0].axis('off')
    axs[1].set_title('2nd')
    # axs[1].axis('off')
    plt.tight_layout()
    plt.savefig('plot_gt_3d_points.png', dpi=300)

    visualize_point_cloud(gt_pts_3d)

    # triangulation_errs = []
    # for i in range(100):
    mapper = SimplifiedMapInitializerCv(K=K)

    ########################################################################################
    ###    Finding homography & estimating initial pose
    ########################################################################################

    (R_hat,t_hat,homography) = mapper.estimate_initial_pose(synth_kp1, synth_kp2)

    # R_delta = R_hat @ R.T
    # cos_theta = (np.trace(R_delta) - 1) / 2.0
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # rot_err = np.degrees(np.arccos(cos_theta))

    # s = (t.T @ t_hat) / (t_hat.T @ t_hat)
    # t_err = np.linalg.norm(s * t_hat - t)

    # print(f"t_err: {t_err} (unit), rot_err: {rot_err} (degrees)")

    
    ########################################################################################
    ###    Initial point triangulation
    ########################################################################################

    pts_3d = mapper.triangulate_initial_point_cloud(synth_kp1, synth_kp2, R_hat, t_hat)
    print(pts_3d.shape, synth_pts_3d.shape)
    triangulation_err = np.sum(np.sqrt(np.sum((pts_3d-synth_pts_3d)**2,axis=1)))/synth_pts_3d.shape[0]

    print(f"Triangulation error: {triangulation_err}")

    visualize_point_cloud(pts_3d.T)
    
    reproj_kp1 = project(pts_3d.T, np.eye(3), np.zeros((3,1)), K).T
    reproj_kp2 = project(pts_3d.T, R, t, K).T

    reproj_err = np.sum(np.sqrt(np.sum((reproj_kp1-synth_kp1)**2, axis=1)))/synth_pts_3d.shape[0]

    print(f"Reprojection error: {reproj_err}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].scatter(reproj_kp1[:,0],reproj_kp1[:,1], s=1)
    axs[1].scatter(reproj_kp2[:,0],reproj_kp2[:,1], s=1)
    axs[0].set_xlim(0, W)
    axs[0].set_ylim(0, H)
    axs[0].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(0, W)
    axs[1].set_ylim(0, H)
    axs[1].set_aspect('equal', adjustable='box')
    axs[0].set_title('1st')
    # axs[0].axis('off')
    axs[1].set_title('2nd')
    # axs[1].axis('off')
    plt.tight_layout()
    plt.savefig('plot_reproj_3d_points.png', dpi=300)

    return

    ########################################################################################
    ###    Real data
    ########################################################################################


    img1 = cv2.imread("/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636750863555584.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/home/andrewhc/Datasets/MH01/mav0/cam0/data/1403636751663555584.png", cv2.IMREAD_GRAYSCALE)

    kp1, kp2 = match_sift_keypoints_and_save_vis(img1, img2)

    # TODO: might want to find the actually fx fy values
    H, W = img1.shape
    K=np.array([
        [3024,  0.0, W/2],
        [ 0.0, 3024, H/2],
        [ 0.0,  0.0, 1.0],
    ])

    mapper = SimplifiedMapInitializerCv(K=K)


    ########################################################################################
    ###    Finding homography & estimating initial pose
    ########################################################################################

    (R,t,H) = mapper.estimate_initial_pose(kp1, kp2)

    # visualize image stitching on real data
    stitched = stitch_images(img1, img2, H)
    cv2.imwrite("H_stitched.png", stitched)

    
    ########################################################################################
    ###    Initial point triangulation
    ########################################################################################

    pts_3d = mapper.triangulate_initial_point_cloud(kp1, kp2, R, t)
    # visualize triangulated 3D points on real data
    visualize_point_cloud(pts_3d)


if __name__ == "__main__":
    main()
