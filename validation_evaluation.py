from noise2seg.models import Noise2Seg, Noise2SegConfig
from skimage import io
import csv
import numpy as np
import pickle
import os
from os.path import join, exists
from os import makedirs as mkdir
from noise2seg.utils.seg_utils import *
from noise2seg.utils.compute_precision_threshold import measure_precision, measure_seg
import argparse
import json


def main():
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    parser = argparse.ArgumentParser(description="Noise2Seg headless score-on-validation-data-script.")
    parser.add_argument('--temp_conf')

    args = parser.parse_args()

    with open(args.temp_conf) as f:
        conf = json.load(f)


    # load data
    trainval_data = np.load(conf['train_data_path'])
    val_images = trainval_data['X_val'].astype(np.float32)
    val_masks = trainval_data['Y_val']

    print("Shape of val_images: ", val_images.shape, ", Shape of val_masks: ", val_masks.shape)

    print("Validation Data \n..................")
    X_val, Y_val_masks = val_images, val_masks

    # one-hot-encoding
    X_val = X_val[...,np.newaxis]
    Y_val = convert_to_oneHot(Y_val_masks)
    print("Shape of validation images: ", X_val.shape, ", Shape of validation masks: ", Y_val.shape)

    # load model

    n2s_model = Noise2Seg(None, conf['model_name'], conf['basedir'])

    # compute AP results
    ap_threshold, validation_ap_score = n2s_model.optimize_thresholds(val_images, Y_val_masks, measure=measure_precision())
    print("Average precision over all validation images at IOU = 0.5 with threshold = {}: ".format(ap_threshold), validation_ap_score)

    # use ap-threshold to compute SEG-scores
    predicted_ap_seg_images, ap_seg_result = n2s_model.predict_label_masks(val_images, Y_val_masks, ap_threshold,
                                                                          measure=measure_seg())


    print("SEG score over all validation images at IOU = 0.5 with ap-threshold = {}: ".format(ap_threshold), ap_seg_result)


    
    # compute SEG results
    seg_threshold, validation_seg_score = n2s_model.optimize_thresholds(val_images, Y_val_masks, measure=measure_seg())
    print("SEG over all validation images at IOU = 0.5 with threshold = {}: ".format(seg_threshold), validation_seg_score)

    with open(join(conf['basedir'], "validation_scores.csv"), mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['AP', validation_ap_score])
        writer.writerow(['SEG', validation_seg_score])
        writer.writerow(['SEG optimized for AP', ap_seg_result])


if __name__=="__main__":
    main()
