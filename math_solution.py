import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    def compute_projection_matrix(K, R_cw, C):
        R_wc = R_cw.T
        t = -R_wc @ C.reshape(-1, 1)
        P = K @ np.hstack((R_wc, t))
        return P

    P1 = compute_projection_matrix(camera_matrix, camera_rotation1, camera_position1)
    P2 = compute_projection_matrix(camera_matrix, camera_rotation2, camera_position2)

    num_points = image_points1.shape[0]
    points_3d = []

    for i in range(num_points):
        p1 = image_points1[i]
        p2 = image_points2[i]

        A = np.array([
            p1[0] * P1[2, :] - P1[0, :],
            p1[1] * P1[2, :] - P1[1, :],
            p2[0] * P2[2, :] - P2[0, :],
            p2[1] * P2[2, :] - P2[1, :]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points_3d.append(X[:3])

    return np.array(points_3d)
