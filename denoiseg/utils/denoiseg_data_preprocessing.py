import numpy as np
from denoiseg.utils.misc_utils import shuffle_train_data


def generate_patches_from_list(data,
                               masks,
                               num_patches_per_img=None,
                               shape=(256, 256),
                               augment=True,
                               shuffle=False, seed=1):
    """
    Extracts patches from 'list_data', which is a list of images, and returns them in a 'numpy-array'. The images
    can have different dimensionality.
    Parameters
    ----------
    data                : list(array(float))
                          List of images

    masks               : list(array(float))
                          List of masks

    axes                : str
                          Possible dimesions include S(number of samples), ZYX(dimesions of a single sample),
                          C(channel, can be singleton dimension). E.g., SYXC in case of 2D data with S samples of shape YX
    num_patches_per_img : int, optional(default=None)
                          Number of patches to extract per image. If 'None', as many patches as fit i nto the
                          dimensions are extracted.
    shape               : tuple(int), optional(default=(256, 256))
                          Shape of the extracted patches.

    augment             : bool, optional(default=True)
                          Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
    shuffle             : bool, optional(default=False)
                          Shuffles extracted patches across all given images (data).
    Returns
    -------
    patches : array(float)
              Numpy-Array with the patches. The dimensions are 'SZYXC' or 'SYXC'
    """
    image_patches, mask_patches = [], []
    assert len(data) == len(masks)

    for img, mask in zip(data, masks):
        for s in range(img.shape[0]):
            p = generate_patches(img[s][np.newaxis], num_patches=num_patches_per_img, shape=shape,
                                 augment=augment)
            m = generate_patches(mask[s][np.newaxis], num_patches=num_patches_per_img, shape=shape,
                                 augment=augment)

            image_patches.append(p)
            mask_patches.append(m)

    train_images = np.concatenate(image_patches, axis=0)
    train_masks = np.concatenate(mask_patches, axis=0)

    if shuffle:
        train_images, train_masks = shuffle_train_data(train_images, train_masks, random_seed=seed)

    return train_images, train_masks


def generate_patches(data, num_patches=None, shape=(256, 256), augment=True, shuffle=False):
    """
    Extracts patches from 'data'. The patches can be augmented, which means they get rotated three times
    in XY-Plane and flipped along the X-Axis. Augmentation leads to an eight-fold increase in training data.
    Parameters
    ----------
    data        : list(array(float))
                  List of images with dimensions 'SZYX' or 'SYX' with optional C
    axes        : str
                  Possible dimesions include S(number of samples), ZYX(dimesions of a single sample),
                  C(channel, can be singleton dimension). E.g., SYXC in case of 2D data with S samples of shape YX
    num_patches : int, optional(default=None)
                  Number of patches to extract per image. If 'None', as many patches as fit i nto the
                  dimensions are extracted.
    shape       : tuple(int), optional(default=(256, 256))
                  Shape of the extracted patches.
    augment     : bool, optional(default=True)
                  Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
    Returns
    -------
    patches : array(float)
              Numpy-Array containing all patches (randomly shuffled along S-dimension).
              The dimensions are 'SZYXC' or 'SYXC'
    """

    patches = extract_patches(data, num_patches=num_patches, shape=shape)
    if shape[-2] == shape[-1]:
        if augment:
            patches = augment_patches(patches=patches)
    else:
        if augment:
            print("XY-Plane is not square. Omit augmentation!")
    if shuffle:
        np.random.shuffle(patches)

    return patches


def extract_patches(data, num_patches=None, shape=(256, 256)):
    n_dims = len(shape)
    if num_patches == None:
        patches = []
        if n_dims == 2:
            if data.shape[1] > shape[0] and data.shape[2] > shape[1]:
                for y in range(0, data.shape[1] - shape[0] + 1, shape[0]):
                    for x in range(0, data.shape[2] - shape[1] + 1, shape[1]):
                        patches.append(data[:, y:y + shape[0], x:x + shape[1]])

                return np.concatenate(patches)
            elif data.shape[1] == shape[0] and data.shape[2] == shape[1]:
                return data
            else:
                print('Incorrect shape')
        elif n_dims == 3:
            target = int((max(16, 2 ** np.ceil(np.log2(data.shape[1])))))
            pad = target - data.shape[1]
            data = np.pad(data, (int(np.ceil(pad / 2)), int(np.floor(pad / 2))), 'constant')
            if data.shape[1] >= shape[0] and data.shape[2] >= shape[1] and data.shape[3] >= shape[2]:
                for z in range(0, data.shape[1] - shape[0] + 1, shape[0]):
                    for y in range(0, data.shape[2] - shape[1] + 1, shape[1]):
                        for x in range(0, data.shape[3] - shape[2] + 1, shape[2]):
                            patches.append(data[:, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

                return np.concatenate(patches)
            elif data.shape[1] == shape[0] and data.shape[2] == shape[1] and data.shape[3] == shape[2]:
                return data
            else:
                print('Incorrect shape')
        else:
            print('Not implemented for more than 4 dimensional (ZYXC) data.')
    else:
        patches = []
        if n_dims == 2:
            for i in range(num_patches):
                y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
                                                                                             data.shape[
                                                                                                 2] - shape[
                                                                                                 1] + 1)
                patches.append(data[0, y:y + shape[0], x:x + shape[1]])

            if len(patches) > 1:
                return np.stack(patches)
            else:
                return np.array(patches)[np.newaxis]
        elif n_dims == 3:
            for i in range(num_patches):
                z, y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
                                                                                                data.shape[
                                                                                                    2] - shape[
                                                                                                    1] + 1), np.random.randint(
                    0, data.shape[3] - shape[2] + 1)
                patches.append(data[0, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

            if len(patches) > 1:
                return np.stack(patches)
            else:
                return np.array(patches)[np.newaxis]
        else:
            print('Not implemented for more than 4 dimensional (ZYXC) data.')


def augment_patches(patches):
    if len(patches.shape[1:]) == 2:
        augmented = np.concatenate((patches,
                                    np.rot90(patches, k=1, axes=(1, 2)),
                                    np.rot90(patches, k=2, axes=(1, 2)),
                                    np.rot90(patches, k=3, axes=(1, 2))))
    elif len(patches.shape[1:]) == 3:
        augmented = np.concatenate((patches,
                                    np.rot90(patches, k=1, axes=(2, 3)),
                                    np.rot90(patches, k=2, axes=(2, 3)),
                                    np.rot90(patches, k=3, axes=(2, 3))))

    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented