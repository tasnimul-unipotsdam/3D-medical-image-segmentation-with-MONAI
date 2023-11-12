import numpy as np
import monai
from surface_distance.metrics import compute_surface_distances, compute_robust_hausdorff


def dice_metric(predicted, target):
    dice_value = monai.losses.DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target)
    return value


def make_one_hot(label):
    result = [label == 0, label == 1, label == 2]

    one_hot = np.stack(result, axis=1).astype(float)

    return one_hot


def robust_hausdorff(m_gt, m_pr, percentage=100):
    m_gt = m_gt.astype(bool)
    # print(m_gt.shape)
    m_gt = np.resize(m_gt, (m_gt.shape[1], m_gt.shape[2], m_gt.shape[3]))

    m_pr = m_pr.astype(bool)
    m_pr = np.resize(m_pr, (m_pr.shape[1], m_pr.shape[2], m_pr.shape[3]))

    surface_distance = compute_surface_distances(m_gt, m_pr, spacing_mm=(1, 1, 1))
    hausdorff_distance = compute_robust_hausdorff(surface_distance, percentage)

    return hausdorff_distance


def dice_coefficient(pred, target, smooth=1.):
    iflat = pred.flatten()
    tflat = target.flatten()

    intersection = np.sum(iflat * tflat)

    A_sum = np.sum(iflat)
    B_sum = np.sum(tflat)

    score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return score


def ECE(conf, acc, n_bins=5):
    """
    acc_bm = sigms 1(\hat{y}_i==y_i)/ |b_m|
    conf_bm= sigma \hat{pi} / |b_m|

    acc_bm == conf_bm


    """

    # print(f'conf: {conf.shape}')
    # print(f'acc:{acc.shape}')

    acc_list = []
    conf_list = []

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_lowers = bin_boundaries[:-1]

    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    bin_counter = 0

    avg_confs_in_bins = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        in_bin = np.logical_and(conf > bin_lower, conf <= bin_upper)

        # |B_m | /n
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:

            # acc_in_bin : |acc_m|/|B_m|

            acc_in_bin = np.mean(acc[in_bin])

            # avg_conf_in_bin: |conf_m|/|B_m|

            avg_conf_in_bin = np.mean(conf[in_bin])

            delta = avg_conf_in_bin - acc_in_bin

            avg_confs_in_bins.append(delta)

            acc_list.append(acc_in_bin)

            conf_list.append(avg_conf_in_bin)

            # |acc_m - conf_m|* |B_m|/n
            ece += np.abs(delta) * prop_in_bin

        else:

            avg_confs_in_bins.append(None)

        bin_counter += 1

        # For reliability diagrams, also need to return these:
        # return ece, bin_lowers, avg_confs_in_bins

    return ece, acc_list, conf_list
