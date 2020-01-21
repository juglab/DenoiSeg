import numpy as np
from skimage.segmentation import find_boundaries


def convert_to_oneHot(data):
    """
    Converts labelled images (`data`) to one-hot encoding.

    Parameters
    ----------
    data : array(int)
        Array of lablelled images.
    Returns
    -------
    data_oneHot : array(int)
        Array of one-hot encoded images.
    """
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
    return data_oneHot


def add_boundary_label(lbl, dtype=np.uint16):
    """
    Find boundary labels for a labelled image.
    Parameters
    ----------
    lbl : array(int)
         lbl is an integer label image (not binarized).
    Returns
    -------
    res : array(int)
        res is an integer label image with boundary encoded as 2.
    """

    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res


def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


def normalize(img, mean, std):
    """
    Mean-Std Normalization.
    Parameters
    ----------
    img : array(float)
        Array of source images.
    mean : float
        mean intensity of images.
    std: float
        standard deviation of intensity of images.
    Returns
    -------
    (img - mean)/std: array(float)
       Normalized images
    """
    return (img - mean) / std


def denormalize(img, mean, std):
    """
    Mean-Std De-Normalization.

    Parameters
    ----------
    img : array(float)
        Array of source images.
    mean : float
        mean intensity of images.
    std: float
        standard deviation of intensity of images.

    Returns
    -------
    img * std + mean: array(float)
        De-normalized images

    """
    return (img * std) + mean


def fractionate_train_data(X_train, Y_train, fraction):
    """
    Fractionates training data according to the specified `fraction`.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    fraction: float (between 0 and 100)
        fraction of training images.

    Returns
    -------
    X_train : array(float)
        Fractionated array of source images.
    Y_train : float
        Fractionated array of label images.
    """
    train_frac = int(np.round((fraction / 100) * X_train.shape[0]))
    X_train = X_train[:train_frac]
    Y_train = Y_train[:train_frac]

    return X_train, Y_train
