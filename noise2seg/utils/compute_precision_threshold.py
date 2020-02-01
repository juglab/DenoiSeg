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

def measure_precision(iou=0.5, partial_dataset=False):
    def precision(lab_gt, lab, iou=iou, partial_dataset=partial_dataset):
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

    return precision


def matching_overlap(psg, fractions=(0.5,0.5)):
    """
    create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    afrac, bfrac = fractions
    tmp = np.sum(psg, axis=1, keepdims=True)
    m0 = np.where(tmp==0,0,psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp==0,0,psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype('bool')
    return matching


def measure_seg(partial_dataset=False):
    def seg(lab_gt, lab, partial_dataset=partial_dataset):
        """
        calculate seg from pixel_sharing_bipartite
        seg is the average conditional-iou across ground truth cells
        conditional-iou gives zero if not in matching
        ----
        calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
        for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
        IoU as low as 1/2 that don't match, and thus have CIoU = 0.
        """
        psg = pixel_sharing_bipartite(lab_gt, lab)
        iou = intersection_over_union(psg)
        matching = matching_overlap(psg, fractions=(0.5, 0))
        matching[0, :] = False
        matching[:, 0] = False
        n_gt = len(set(np.unique(lab_gt)) - {0})
        n_matched = iou[matching].sum()
        if partial_dataset:
            return n_matched, n_gt
        else:
            return n_matched / n_gt

    return seg


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


def compute_labels(prediction, threshold):
    prediction_exp = np.exp(prediction[..., 1:])
    prediction_softmax = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
    prediction_fg = prediction_softmax[..., 1]
    pred_thresholded = prediction_fg > threshold
    labels, _ = ndimage.label(pred_thresholded)
    return labels


def compute_threshold(X_val, Y_val, model, measure=measure_precision()):
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
        score = 0
        for idx in range(X_val.shape[0]):
            img, gt = X_val[idx], Y_val[idx]
            prediction = model.predict(img, axes='YX')
            labels = compute_labels(prediction, ts)
            tmp_score = measure(gt, labels)
            if not np.isnan(tmp_score):
                score += tmp_score

        score /= float(X_val.shape[0])
        precision_scores.append((ts, score))
        print('Score for threshold =', "{:.2f}".format(ts), 'is', "{:.4f}".format(score))

    sorted_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
    computed_threshold = sorted_score[0]
    best_score = sorted_score[1]
    return computed_threshold, best_score
