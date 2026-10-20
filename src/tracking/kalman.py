"""
FAST-PLATE-OCR-VIETNAMESE: Kalman Filter for Bounding Box Tracking
Bộ lọc Kalman 8 chiều theo dõi vị trí và vận tốc bounding box trong không gian 2D:
State: [cx, cy, a, h, vx, vy, va, vh]
trong đó cx, cy là tọa độ tâm, a = w/h là tỷ lệ khung hình, h là chiều cao.
"""

import numpy as np
from typing import Tuple

Tuple_np = Tuple[np.ndarray, np.ndarray]


class KalmanFilter:
    """
    Bộ lọc Kalman tuyến tính theo dõi chuyển động bounding box với giả định vận tốc không đổi (Constant Velocity Model).
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Ma trận chuyển trạng thái (State Transition Matrix F)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Ma trận quan sát (Observation Matrix H)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Trọng số nhiễu
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple_np:
        """Khởi tạo trạng thái ban đầu từ quan sát đầu tiên [cx, cy, a, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple_np:
        """Dự đoán trạng thái kế tiếp."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple_np:
        """Chiếu không gian trạng thái vào không gian quan sát."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple_np:
        """Cập nhật trạng thái khi có quan sát mới."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy_cho_factor(projected_cov)
        kalman_gain = scipy_cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def scipy_cho_factor(matrix):
    """Phân rã Cholesky nội bộ tối ưu bằng NumPy không phụ thuộc scipy."""
    return np.linalg.cholesky(matrix), True


def scipy_cho_solve(chol_tuple, b):
    """Giải hệ phương trình Cholesky bằng NumPy."""
    L, _ = chol_tuple
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x


Tuple_np = tuple
