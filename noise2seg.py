from noise2seg.models import Noise2Seg, Noise2SegConfig
from skimage import io
import csv
import numpy as np
import pickle
import os
from os.path import join, exists
from os import makedirs as mkdir
from noise2seg.utils.misc_utils import shuffle_train_data, augment_data
from noise2seg.utils.seg_utils import *
from noise2seg.utils.compute_precision_threshold import measure_precision, measure_seg
import argparse
import json


def main():
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    parser = argparse.ArgumentParser(description="Noise2Seg headless train-predict-score-script.")
    parser.add_argument('--exp_conf')

    args = parser.parse_args()

    with open(args.exp_conf) as f:
        conf = json.load(f)


    # load data
    trainval_data = np.load(conf['train_data_path'])
    test_data = np.load(conf['test_data_path'], allow_pickle=True)
    train_images = trainval_data['X_train'].astype(np.float32)
    val_images = trainval_data['X_val'].astype(np.float32)
    test_images = test_data['X_test']

    train_masks = trainval_data['Y_train']
    val_masks = trainval_data['Y_val']
    test_masks = test_data['Y_test']

    print("Shape of train_images: ", train_images.shape, ", Shape of train_masks: ", train_masks.shape)
    print("Shape of val_images: ", val_images.shape, ", Shape of val_masks: ", val_masks.shape)
    print("Shape of test_images: ", test_images.shape, ", Shape of test_masks: ", test_masks.shape)

    # shuffling, fractioning, augmenting
    assert 0 <conf['fraction']<= 100, "Fraction should be between 0 and 100"

    X_shuffled, Y_shuffled = shuffle_train_data(train_images, train_masks, random_seed = conf['random_seed'])
    X_frac, Y_frac = zero_out_train_data(X_shuffled, Y_shuffled, fraction = conf['fraction'])
    print("Training Data \n..................")
    X, Y_train_masks = augment_data(X_frac, Y_frac)
    print("\n")
    print("Validation Data \n..................")
    X_val, Y_val_masks = val_images, val_masks

    # one-hot-encoding
    X = X[...,np.newaxis]
    Y = convert_to_oneHot(Y_train_masks)
    X_val = X_val[...,np.newaxis]
    Y_val = convert_to_oneHot(Y_val_masks)
    print("Shape of train images: ", X.shape, ", Shape of train masks: ", Y.shape)
    print("Shape of validation images: ", X_val.shape, ", Shape of validation masks: ", Y_val.shape)

    # create model
    train_steps_per_epoch = max(100, min(int(X.shape[0]/conf['train_batch_size']), 400))
    print("train_steps_per_epoch =", train_steps_per_epoch)
    n2s_conf = Noise2SegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights = [1.0,1.0,5.0],
                       train_steps_per_epoch=train_steps_per_epoch, train_epochs=conf['train_epochs'], train_loss='noise2seg', batch_norm=True,
                       train_batch_size=conf['train_batch_size'], unet_n_first = 32, unet_n_depth=conf['unet_n_depth'],
                               n2s_alpha=conf['n2s_alpha'],
                              train_tensorboard=False,
                               train_reduce_lr={"patience" : 10,
                                                "min_delta" : 0.00001,
                                                "factor" : 0.75,
                                                "min_lr" : 0.0000125
                                                  })

    vars(n2s_conf)
    n2s_model = Noise2Seg(n2s_conf, conf['model_name'], conf['basedir'])

    # train
    history = n2s_model.train(X, Y, (X_val, Y_val))
    with open(join(conf['basedir'], conf['model_name'], 'history.hist'), 'wb') as p:
        pickle.dump(history, p)

    # load model
    n2s_model = Noise2Seg(None, conf['model_name'], conf['basedir'])

    # compute AP results
    ap_threshold, validation_ap_score = n2s_model.optimize_thresholds(val_images, Y_val_masks, measure=measure_precision())
    predicted_ap_images, precision_result = n2s_model.predict_label_masks(test_images, test_masks, ap_threshold, measure=measure_precision())
    print("Average precision over all test images at IOU = 0.5 with threshold = {}: ".format(ap_threshold), precision_result)

    ap_path = join(conf['basedir'], "AP")
    if not exists(ap_path):
        mkdir(ap_path)
    for i in range(len(predicted_ap_images)):
        io.imsave(join(ap_path, 'mask' + str(i).zfill(3) + '.tif'), predicted_ap_images[i].astype(np.int16))

    # use ap-threshold to compute SEG-scores
    validation_ap_seg_images, validation_ap_seg_score = n2s_model.predict_label_masks(val_images, Y_val_masks, ap_threshold,
                                                                          measure=measure_seg())
    predicted_ap_seg_images, ap_seg_result = n2s_model.predict_label_masks(test_images, test_masks, ap_threshold,
                                                                          measure=measure_seg())
    print("SEG score over all test images at IOU = 0.5 with ap-threshold = {}: ".format(ap_threshold),
          ap_seg_result)

    ap_seg_path = join(conf['basedir'], "SEG_AP-Threshold")
    if not exists(ap_seg_path):
        mkdir(ap_seg_path)
    for i in range(len(predicted_ap_seg_images)):
        io.imsave(join(ap_seg_path, 'mask' + str(i).zfill(3) + '.tif'), predicted_ap_seg_images[i].astype(np.int16))

    # compute SEG results
    seg_threshold, validation_seg_score = n2s_model.optimize_thresholds(val_images, Y_val_masks, measure=measure_seg())
    predicted_seg_images, seg_result = n2s_model.predict_label_masks(test_images, test_masks, seg_threshold, measure=measure_seg())
    print("SEG over all test images at IOU = 0.5 with threshold = {}: ".format(seg_threshold), seg_result)

    seg_path = join(conf['basedir'], "SEG")
    if not exists(seg_path):
        mkdir(seg_path)
    for i in range(len(predicted_seg_images)):
        io.imsave(join(seg_path, 'mask' + str(i).zfill(3) + '.tif'), predicted_seg_images[i].astype(np.int16))

    with open(join(conf['basedir'], "scores.csv"), mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['AP', precision_result])
        writer.writerow(['SEG', seg_result])
        writer.writerow(['SEG_AP-Threshold', ap_seg_result])
        
    with open(join(conf['basedir'], "validation_scores.csv"), mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['AP', validation_ap_score])
        writer.writerow(['SEG', validation_seg_score])
        writer.writerow(['SEG_AP-Threshold', validation_ap_seg_score ])


if __name__=="__main__":
    main()
