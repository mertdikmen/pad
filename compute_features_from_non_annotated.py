import numpy as np
import pylab as plt
import os
import sys
from scipy.misc import imsave
import vivid

from config import ConfigOpt
from opt import options

from feature_source import *
from libsvm_helper import *

import time

FALSE_POS_THRESHOLD = 0

np.random.seed(1905)

config_sources = options.config_source.split(',')

def process_subrange(n_data_splits, data_split_ind):
    config_opts = [ConfigOpt(cs) for cs in config_sources]

    num_images = len(config_opts[0].non_annotation_image_inds[options.set_type])

    split_inds = np.linspace(0, num_images, num=n_data_splits + 1)

    data_split_start = int(split_inds[data_split_ind])
    data_split_end = int(split_inds[data_split_ind + 1])

    num_split_images = data_split_end - data_split_start

    non_boot_samples_per_im = config_opts[0].training_samples_per_im

    feature_dims = np.array([cos.feature.feature_dim for cos in config_opts])
    feature_dims_cs = np.cumsum(feature_dims)

    false_positives = np.empty((0, feature_dims.sum()), dtype='float32')

    total_non_boot_samples = num_split_images * non_boot_samples_per_im

    feature_str = ''
    for cos in config_opts:
        feature_str += cos.feature_str

    if options.bootstrap > 0:
        svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                      "model1.%ssvm" % feature_str)

        lm = read_liblinear_model(svm_model_file)
        w = lm.w[:-1]
        b = lm.w[-1] * lm.bias

    if options.bootstrap > 1:
        svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                      "modelb.%ssvm" % feature_str)

        lmb = read_liblinear_model(svm_model_file)

        wb = lmb.w[:-1]
        bb = lmb.w[-1] * lbm.bias

    fv = vivid.ImageSource(imlist=config_opts[0].image_sets_fp[options.set_type])
    cs = vivid.ConvertedSource(fv,
                               target_type=vivid.cv.CV_32FC3,
                               scale=1.0 / 255.0)
    gv = vivid.GreySource(cs)
    
    feature_sources = []

    for cos in config_opts:
        if cos.feature.fea_type == 'flex':
            clusters = np.load(cos.dictionary_file).reshape(
                                (cos.feature.cell.dictionary_size,
                                 cos.feature.cell.patch_size,
                                 cos.feature.cell.patch_size))
        
            feature_sources.append(FeatureSource(gv, cos.feature, clusters))
        elif cos.feature.fea_type == 'lbp':
            feature_sources.append(FeatureSource(gv, cos.feature))
        elif cos.feature.fea_type == 'uniform':
            feature_sources.append(FeatureSource(gv, cos.feature))
        else:
            raise ValueError("Unknown feature type")

    counter = 0

    all_features = np.zeros((total_non_boot_samples,
                             feature_dims.sum()), dtype='f')

    total_fp = 0
    collected = 0
    for i, image_ind in enumerate(config_opts[0].non_annotation_image_inds[
                            options.set_type][data_split_start:data_split_end]
                            ):

        print("{0} - {1}/{2}: {3}".format(
            data_split_ind,
            i,
            num_split_images,
            config_opts[0].image_sets_fp[options.set_type][image_ind]))

        for fea_i, fs in enumerate(feature_sources):
            fs.init_frame(image_ind)
    
            n_scales = len(fs.scales)
    
            if n_scales == 0:
                continue

        image_samples = np.zeros((n_scales * non_boot_samples_per_im,
                                  feature_dims.sum()),
                                 dtype='float32')

        fea_sum = 0
        scale_sample_ind = 0

        for si in range(n_scales):
            for fs in feature_sources:
                locs, scale = fs.init_scale(si)

            num_y, num_x = locs[0].shape
            n_total = locs[0].size

            random_ind = np.random.permutation(n_total)[
                :non_boot_samples_per_im]

            scale_random_ind_x, scale_random_ind_y = np.meshgrid(
                range(num_x), range(num_y))

            scale_random_ind_x = scale_random_ind_x.flat[random_ind]
            scale_random_ind_y = scale_random_ind_y.flat[random_ind]

            for yind in range(0, num_y, 10):
                ymin = yind
                ymax = min(num_y, yind + 10)

                random_ind_part = np.logical_and(
                    scale_random_ind_y >= ymin,
                    scale_random_ind_y < ymax)

                n_random_ind_part = random_ind_part.sum()

                if n_random_ind_part == 0 and options.bootstrap < 1:
                    continue

                scale_random_ind_x_part = scale_random_ind_x[
                                            random_ind_part]
                scale_random_ind_y_part = scale_random_ind_y[
                                            random_ind_part]

                feas = [fs.get_features_from_scale(
                    ymin=ymin, ymax=ymax,
                    xmin=0, xmax=num_x) for fs in feature_sources]

                feas = np.concatenate(feas, axis=2)

                image_samples[
                    scale_sample_ind : 
                    scale_sample_ind + n_random_ind_part,
                    :] = feas[scale_random_ind_y_part - ymin,
                              scale_random_ind_x_part, :].copy()

                scale_sample_ind += n_random_ind_part

                if options.bootstrap > 0:
                    if options.tfidf == 1:
                        scores = ((feas * tfidf_w[np.newaxis, np.newaxis, :] *
                            w[np.newaxis, np.newaxis, :]).sum(axis=2) + b)
                    else:
                        scores = ((feas *
                            w[np.newaxis, np.newaxis, :]).sum(axis=2) + b)

                    false_fp = np.nonzero(scores > FALSE_POS_THRESHOLD)

                    num_fp = false_fp[0].size

                    false_positives = np.vstack((
                                        false_positives,
                                        feas[false_fp[0], false_fp[1], :]))

                    total_fp += num_fp
                    print("Scale: {0:.2f}, FP: {1}".format(scale, num_fp))

        image_random_ind = np.random.permutation(scale_sample_ind)[:
                            non_boot_samples_per_im]

        all_features[collected:
                     collected + image_random_ind.size, :] = (
           image_samples[image_random_ind, :].copy())

        collected += image_random_ind.size

    all_features = all_features[:collected]

    bootstrap_str = ""
    if options.bootstrap:
        bootstrap_str = "b."

    target_file = os.path.join(
        config_opts[0].feature_path,
        "non_annotation_features_%s.%s%s%03d:%03d.npy" % (
            options.set_type,
            feature_str,
            bootstrap_str,
            data_split_ind,
            n_data_splits))

    if options.bootstrap:
        all_features = np.vstack((all_features[:collected], false_positives))
        collected = all_features.shape[0]

    print("Saving to: {0}".format(target_file))

    assert(np.all(
        (all_features * all_features).sum(axis=1) 
        >= 0.9999))

    np.save(target_file, all_features)

def process_subrange_tup(args):
    return process_subrange(*args)

if __name__ == '__main__':
    if options.data_split_ind != -1:
        process_subrange(options.data_split, options.data_split_ind)
    else:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(n_cpus)

        pool.map(process_subrange_tup, zip(
            [options.data_split for x in range(options.data_split)],
            range(options.data_split)))

