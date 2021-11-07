import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt

def sigmoid(x):
    return 1/(1+np.exp(-x))


def bce(true, pred):
    true = np.ravel(true)
    pred = np.ravel(pred)

    true = np.clip(true, 1e-6, 1-1e-6)
    pred = np.clip(pred, 1e-6, 1-1e-6)

    return -np.mean(true * np.log(pred) + (1-true) * np.log(1-pred))


def dice(true, pred):
    true = np.ravel(true)
    pred = np.ravel(pred)

    return (2*np.dot(true, pred)+0.0001)/((np.linalg.norm(true)**2 + np.linalg.norm(pred)**2) + 0.0001)


def hausdorff(true, pred, pix_dim=None, method='max', true_dist_map=None, pred_dist_map=None, return_all=True):
    if pix_dim is None:
        pix_dim = [1,1,1]

    true = true > 0
    pred = pred > 0

    if true_dist_map is None:
        d_from_true = distance_transform_edt(~true, pix_dim)
    else:
        d_from_true = true_dist_map

    if pred_dist_map is None:
        d_from_pred = distance_transform_edt(~pred, pix_dim)
    else:
        d_from_pred = pred_dist_map

    true_points = np.where(true)
    pred_points = np.where(pred)
    true_to_pred = d_from_pred[true_points]
    pred_to_true = d_from_true[pred_points]
    nb_true_points = len(true_points[0])
    nb_pred_points = len(pred_points[0])

    if nb_true_points == 0 or nb_pred_points == 0:
        if return_all:
            return np.nan, np.nan, np.nan
        else:
            return np.nan
    else:
        t_to_p = np.sum(true_to_pred)/nb_true_points
        p_to_t = np.sum(pred_to_true)/nb_pred_points

        if method == 'max':
            if return_all:
                return max(t_to_p, p_to_t), t_to_p, p_to_t
            else:
                return max(t_to_p, p_to_t)
        elif method == 'mean':
            if return_all:
                return 0.5*(t_to_p + p_to_t), t_to_p, p_to_t
            else:
                return 0.5*(t_to_p + p_to_t)
        else:
            raise ValueError


def roc(true, pred, return_sens_spec=False):

    true = np.ravel(true)
    pred = np.ravel(pred)

    true_pos = sum(true*pred)
    false_pos = sum(pred*(1-true))
    false_neg = sum((1-pred)*true)
    true_neg = sum((1-pred)*(1-true))

    sens = (true_pos) / (true_pos + false_neg)
    spec = (true_neg) / (false_pos + true_neg)

    if return_sens_spec:
        return sens, spec
    else:
        return true_pos, true_neg, false_pos, false_neg

