import math
import numpy as np
from PIL import Image


def compute_h(p1, p2):
    # initialize parameters
    N, _ = p1.shape
    p2_homogeneous = np.concatenate([p2, np.ones((N, 1))], axis=1)

    # consturct A
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2 * i, :] = np.array(
            np.concatenate([p2_homogeneous[i, :], [0, 0, 0], -p1[i, 0] * p2_homogeneous[i, :]], axis=0)
        )
        A[2 * i + 1, :] = np.array(
            np.concatenate([[0, 0, 0], p2_homogeneous[i, :], -p1[i, 1] * p2_homogeneous[i, :]], axis=0)
        )

    # find h which minimizes L2 norm of Ah
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]

    # reshape h to 3x3
    H = h.reshape((3, 3))

    return H


def compute_h_norm(p1, p2):
    # initialize parameters
    p1x_max, p1y_max = np.max(p1, axis=0)
    p2x_max, p2y_max = np.max(p2, axis=0)
    p1_norm_matrix = np.diag((p1x_max, p1y_max, 1))
    p2_norm_matrix = np.diag((p2x_max, p2y_max, 1))

    # normalize vectors
    normalized_p1 = p1 / (p1x_max, p1y_max)
    normalized_p2 = p2 / (p2x_max, p2y_max)

    # compute homogrpahy of normalized vectors
    normalized_H = compute_h(normalized_p1, normalized_p2)

    # reconstruct original homography
    H = p1_norm_matrix @ normalized_H @ np.linalg.inv(p2_norm_matrix)

    return H


def warp_image(igs_in, igs_ref, H):
    # initialize parameters
    left_pad = 0
    right_pad = 0
    up_pad = 0
    down_pad = 0
    h, w, _ = igs_ref.shape
    ref_h, ref_w, _ = igs_ref.shape
    h_warp = h + up_pad + down_pad
    w_warp = w + left_pad + right_pad
    igs_warp = np.zeros((h_warp, w_warp, 3))
    igs_merge = igs_ref.copy()

    # bilinear interpolation
    def bilinear_interpolate(img, x, y):
        h, w, _ = img.shape
        x_left, x_right = int(x), int(x) + 1
        y_up, y_down = int(y), int(y) + 1

        if x_left < 0 or x_right >= w or y_up < 0 or y_down >= h:
            return 0

        dx = x - int(x)
        dy = y - int(y)

        return (
            img[y_up, x_left, :] * (1 - dx) * (1 - dy)
            + img[y_up, x_right, :] * dx * (1 - dy)
            + img[y_down, x_left, :] * (1 - dx) * dy
            + img[y_down, x_right, :] * dx * dy
        )

    # compute inverse homography
    H_inverse = np.linalg.inv(H)

    # warp image
    indices = (np.array([[x, y, 1] for y in range(h_warp) for x in range(w_warp)]) - np.array([left_pad, up_pad, 0])).T
    warped_homogeneous = H_inverse @ indices
    warped_x, warped_y, _ = warped_homogeneous / warped_homogeneous[2]
    for i in range(len(indices.T)):
        interpolated_value = bilinear_interpolate(igs_in, warped_x[i], warped_y[i])
        if np.any(interpolated_value != 0):
            igs_merge[up_pad + indices[1, i], left_pad + indices[0, i]] = interpolated_value 
        igs_warp[up_pad + indices[1, i], left_pad + indices[0, i]] = interpolated_value

    return igs_warp, igs_merge