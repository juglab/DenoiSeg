import os
import glob
import shutil
import tifffile
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image


def create_patches(images, masks, size):
    """
    Creates square patches from images and masks of a specified `size`.

    Parameters
    ----------
    images : array(float)
        Array of images.
    masks : array(int)
        Array of labelled images.
    size: int
        Width of the patch
    Returns
    -------
    patchesimages : array(float)
        Array of training patches.
    patchesmasks : array(float)
        Array of labelled training patches.
    """

    patchesimages = image.extract_patches_2d(images, (size, size), max_patches=10, random_state=0)
    patchesmasks = image.extract_patches_2d(masks, (size, size), max_patches=10, random_state=0)
    return patchesimages, patchesmasks


def add_noise(image, sigma):
    mean = 0
    img = np.array(image).astype(np.float32)
    gauss = np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = img + gauss
    return noisy


def crop_3d(img, crop_z, crop_y, crop_x):
    """
    Used to crop 3d volume in 2 dimensions by equal values and separately crop Z dimension
    :param img: Input volume
    :param crop_z: Number of slices in Z dimension
    :param crop_y: Number of pixels to crop from y axis
    :param crop_x: Number of pixels to crop from x axis
    """
    z, y, x = img.shape
    start_z = (z-crop_z) // 2
    start_y = (y // 2) - (crop_y // 2)
    start_x = (x // 2) - (crop_x // 2)
    return img[start_z:start_z + 32, start_y:start_y + crop_y, start_x:start_x + crop_x]


def read_images(path):
    """
    :param path: List of pathlike objects
    :return: Array of images
    """
    images = Parallel(n_jobs=4, prefer='threads')(delayed(tifffile.imread)(p) for p in path)
    return np.stack(images)


def combine_train_test_data(X_train, Y_train, X_test, Y_test):
    """
    Combines train and test data along 0th dimension.

    Parameters
    ----------
    X_train : array(float)
        Array of training images.
    Y_train : float
        Array of labelled training images.
    X_test : array(float)
        Array of test images.
    Y_test : float
        Array of labelled test images.
    Returns
    -------
    X_train_N2V : array(float)
        Combined array of training images.
    Y_train_N2V : array(float)
        Combined array of labelled training images.
    """

    if (X_test.ndim == X_train.ndim):
        if (X_test.shape[1] == X_train.shape[1] and X_test.shape[2] == X_train.shape[2]):
            X_test_patches = X_test
            Y_test_patches = Y_test
    else:
        X_test_patches, Y_test_patches = create_patches(X_test[0], Y_test[0], X_train.shape[1])
        for image_num in range(1, X_test.shape[0]):
            patchesimages, patchesmasks = create_patches(X_test[image_num], Y_test[image_num], X_train.shape[1])
            X_test_patches = np.concatenate((X_test_patches, patchesimages))
            Y_test_patches = np.concatenate((Y_test_patches, patchesmasks))

    X_train_N2V = np.concatenate((X_train, X_test_patches))
    Y_train_N2V = np.concatenate((Y_train, Y_test_patches))

    return X_train_N2V, Y_train_N2V


def split_train_test_data(data_path: str, test_subset=0.2, seed=1000):
    """
        Splits the `train` directory into `test` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_path: string
            Dataset root, without train
        subset: float
            Test percentage
        seed: intege
            Reproducibility constant.
    """

    # Initial location
    images_dir = Path(data_path) / 'train' / 'images'
    masks_dir = Path(data_path) / 'train' / 'masks'

    image_names = sorted(list(images_dir.rglob('*.tif')))
    mask_names = sorted(list(masks_dir.rglob( '*.tif')))
    images_train, images_test, masks_train, masks_test = train_test_split(image_names,
                                                                          mask_names,
                                                                          test_size=test_subset,
                                                                          random_state=seed)
    # Create test dirs
    (Path(data_path) / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(data_path) / 'test' / 'masks').mkdir(parents=True, exist_ok=True)

    for i in range(len(images_test)):
        shutil.move(str(images_test[i]), str(Path(data_path) / 'test' / 'images'))
        shutil.move(str(masks_test[i]), str(Path(data_path) / 'test' / 'masks'))
    print(f'Test Images/Masks saved at {Path(data_path) / "test"}')


def shuffle_train_data(X_train, Y_train, random_seed):
    """
    Shuffles data with seed 1.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train : array(float)
        shuffled array of training images.
    Y_train : array(float)
        Shuffled array of labelled training images.
    """
    np.random.seed(random_seed)
    seed_ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[seed_ind]
    Y_train = Y_train[seed_ind]

    return X_train, Y_train


def augment_data(array, axes: str):
    """
    Augments the data 8-fold by 90 degree rotations and flipping.
    Takes a dimension S.
    """
    # Adapted from DenoiSeg, in order to work with the following order `SZYXC`
    ind_x = axes.find('X')
    ind_y = axes.find('Y')

    # rotations
    _x = array.copy()
    X_rot = [np.rot90(_x, i, (ind_y, ind_x)) for i in range(4)]
    X_rot = np.concatenate(X_rot, axis=0)

    # flip
    X_flip = np.flip(X_rot, axis=ind_y)

    return np.concatenate([X_rot, X_flip], axis=0)

