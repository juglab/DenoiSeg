import numpy as np
from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook


@jit
def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=int)
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
    prediction_fg = prediction[..., 1]
    
    pred_thresholded = prediction_fg > threshold
    labels, _ = ndimage.label(pred_thresholded)

    prediction_binary = np.where(prediction_fg > threshold, np.ones_like(prediction_fg), np.zeros_like(prediction_fg))

    return labels, prediction_binary
