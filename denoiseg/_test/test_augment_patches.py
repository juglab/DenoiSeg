import pytest
import numpy as np

from denoiseg.utils.denoiseg_data_preprocessing import augment_patches


def test_augment_data_simple():
    """
    Test the augmentation of patches for different shapes.
    """
    ################
    # dims (2, 2, 2)
    axes = 'SYX'
    r = np.array([
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]
    ])
    r1 = np.array([
        [[2, 4],
         [1, 3]],
        [[6, 8],
         [5, 7]]
    ])
    r2 = np.array([
        [[4, 3],
         [2, 1]],
        [[8, 7],
         [6, 5]]
    ])
    r3 = np.array([
        [[3, 1],
         [4, 2]],
        [[7, 5],
         [8, 6]]
    ])
    f0 = np.array([
        [[3, 4],
         [1, 2]],
        [[7, 8],
         [5, 6]]
    ])
    f1 = np.array([
        [[1, 3],
         [2, 4]],
        [[5, 7],
         [6, 8]]
    ])
    f2 = np.array([
        [[2, 1],
         [4, 3]],
        [[6, 5],
         [8, 7]]
    ])
    f3 = np.array([
        [[4, 2],
         [3, 1]],
        [[8, 6],
         [7, 5]]
    ])

    # concatenate all rotations and flips
    x_final = np.concatenate([r, r1, r2, r3, f0, f1, f2, f3], axis=0)

    x_aug = augment_patches(r, axes)
    assert x_aug.shape == x_final.shape
    assert (x_aug == x_final).all()

    #######################
    # Add singleton S and Z
    # dims (1, 2, 2, 2)
    axes = 'SZYX'
    r = r[np.newaxis, ...]
    r1 = r1[np.newaxis, ...]
    r2 = r2[np.newaxis, ...]
    r3 = r3[np.newaxis, ...]
    f0 = f0[np.newaxis, ...]
    f1 = f1[np.newaxis, ...]
    f2 = f2[np.newaxis, ...]
    f3 = f3[np.newaxis, ...]
    x_final = np.concatenate([r, r1, r2, r3, f0, f1, f2, f3], axis=0)

    x_aug = augment_patches(r, axes)
    assert x_aug.shape == x_final.shape
    assert (x_aug == x_final).all()

    #######################
    # Add singleton channel
    # dims (1, 2, 2, 2, 1)
    axes = 'SZYXC'
    r = r[..., np.newaxis]
    r1 = r1[..., np.newaxis]
    r2 = r2[..., np.newaxis]
    r3 = r3[..., np.newaxis]
    f0 = f0[..., np.newaxis]
    f1 = f1[..., np.newaxis]
    f2 = f2[..., np.newaxis]
    f3 = f3[..., np.newaxis]
    x_final = np.concatenate([r, r1, r2, r3, f0, f1, f2, f3], axis=0)

    x_aug = augment_patches(r, axes)
    assert x_aug.shape == x_final.shape
    assert (x_aug == x_final).all()

    ###################
    # Add a channel
    # dims (1, 2, 2, 2)
    axes = 'SYXC'
    r = r[:, 0, ...]
    r1 = r1[:, 0, ...]
    r2 = r2[:, 0, ...]
    r3 = r3[:, 0, ...]
    f0 = f0[:, 0, ...]
    f1 = f1[:, 0, ...]
    f2 = f2[:, 0, ...]
    f3 = f3[:, 0, ...]

    r = np.concatenate([r, 10*r], axis=-1)
    r1 = np.concatenate([r1, 10*r1], axis=-1)
    r2 = np.concatenate([r2, 10*r2], axis=-1)
    r3 = np.concatenate([r3, 10*r3], axis=-1)
    f0 = np.concatenate([f0, 10*f0], axis=-1)
    f1 = np.concatenate([f1, 10*f1], axis=-1)
    f2 = np.concatenate([f2, 10*f2], axis=-1)
    f3 = np.concatenate([f3, 10*f3], axis=-1)
    x_final = np.concatenate([r, r1, r2, r3, f0, f1, f2, f3], axis=0)

    x_aug = augment_patches(r, axes)
    assert x_aug.shape == x_final.shape
    assert (x_aug == x_final).all()


@pytest.mark.parametrize('shape, axes', [((1, 16, 16), 'SYX'),
                                         ((8, 16, 16), 'SYX'),
                                         ((1, 10, 16, 16), 'SZYX'),
                                         ((32, 10, 16, 16), 'SZYX'),
                                         ((1, 10, 16, 16, 3), 'SZYXC'),
                                         ((32, 10, 16, 16, 3), 'SZYXC')])
def test_augment_data(shape, axes):
    """
    Test that augmentation runs through with different shapes and random numbers.
    """
    x = np.random.randint(0, 65535, shape, dtype=np.uint16)
    x_aug = augment_patches(x, axes)

    assert x_aug.shape == (x.shape[0] * 8,) + x.shape[1:]
