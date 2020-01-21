import numpy as np
from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook


@jit
def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=np.int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg


def intersection_over_union(psg):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)


def matching_iou(psg, fraction=0.5):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    iou = intersection_over_union(psg)
    matching = iou > 0.5
    matching[:, 0] = False
    matching[0, :] = False
    return matching


def precision(lab_gt, lab, iou=0.5, partial_dataset=False):
    """
    precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
    :Authors:
        Coleman Broaddus
    """
    psg = pixel_sharing_bipartite(lab_gt, lab)
    matching = matching_iou(psg, fraction=iou)
    assert matching.sum(0).max() < 2
    assert matching.sum(1).max() < 2
    n_gt = len(set(np.unique(lab_gt)) - {0})
    n_hyp = len(set(np.unique(lab)) - {0})
    n_matched = matching.sum()
    if partial_dataset:
        return n_matched, (n_gt + n_hyp - n_matched)
    else:
        return n_matched / (n_gt + n_hyp - n_matched)


def isnotebook():
    """
    Checks if code is run in a notebook, which can be useful to determine what sort of progressbar to use.
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/24937408#24937408

    Returns
    -------
    bool
        True if running in notebook else False.

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def compute_threshold(X_val, Y_val, model, mode=None):
    """
    Computes average precision (AP) at different probability thresholds on validation data and returns the best-performing threshold.

    Parameters
    ----------
    X_val : array(float)
        Array of validation images.
    Y_val : array(float)
        Array of validation labels
    model: keras model

    mode: 'none', 'StarDist'
        If `none`, consider a U-net type model, else, considers a `StarDist` type model
    Returns
    -------
    computed_threshold: float
        Best-performing threshold that gives the highest AP.


    """
    print('Computing best threshold: ')
    precision_scores = []
    if (isnotebook()):
        progress_bar = tqdm_notebook
    else:
        progress_bar = tqdm
    for ts in progress_bar(np.linspace(0.1, 1, 19)):
        precision_score = 0
        for idx in range(X_val.shape[0]):
            img, gt = X_val[idx], Y_val[idx]
            if (mode == "StarDist"):
                labels, _ = model.predict_instances(img, prob_thresh=ts)
            else:
                prediction = model.predict(img, axes='YX')
                prediction_exp = np.exp(prediction[..., :])
                prediction_precision = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_fg = prediction_precision[..., 1]
                pred_thresholded = prediction_fg > ts
                labels, _ = ndimage.label(pred_thresholded)

            tmp_score = precision(gt, labels)
            if not np.isnan(tmp_score):
                precision_score += tmp_score

        precision_score /= float(X_val.shape[0])
        precision_scores.append((ts, precision_score))
        print('Precision-Score for threshold =', "{:.2f}".format(ts), 'is', "{:.4f}".format(precision_score))

    best_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
    computed_threshold = best_score[0]
    return computed_threshold
