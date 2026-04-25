"""Contains function for generating label distribution."""

import numpy as np


def genLD(label, sigma, loss, class_num):
    """Generate label distribution."""

    label = np.array(label, dtype=np.float32)
    label_set = np.arange(class_num, dtype=np.float32)

    if loss == "klloss":

        ld_num = len(label_set)

        # reshape cho đúng broadcast
        label_set = label_set.reshape(ld_num, 1)
        label = label.reshape(1, -1)

        # Gaussian
        dif = label_set - label

        # tránh sigma = 0
        sigma = max(float(sigma), 1e-6)

        ld = (
            1.0
            / (np.sqrt(2.0 * np.pi) * sigma)
            * np.exp(- (dif ** 2) / (2.0 * sigma ** 2))
        )

        # ===== FIX QUAN TRỌNG =====
        sum_ld = np.sum(ld, axis=0, keepdims=True)

        # tránh chia 0
        sum_ld[sum_ld == 0] = 1e-8

        ld = ld / sum_ld

        return ld.T