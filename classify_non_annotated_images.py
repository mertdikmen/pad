import numpy as np
import pylab as plt
import os
import vivid

from libsvm_helper import *

from feature_source import *

from config import ConfigOpt
from opt import options

config_sources = options.config_source.split(',')

config_opts = [ConfigOpt(cs) for cs in config_sources]

feature_str = ''
for cos in config_opts:
    feature_str += cos.feature_str


def classify_subrange(n_data_splits, data_split_ind):

    num_images = len(config_opts[0].non_annotation_image_inds[options.set_type])

    split_inds = np.linspace(0, num_images, num=n_data_splits + 1)

    data_split_start = int(split_inds[data_split_ind])
    data_split_end = int(split_inds[data_split_ind + 1])

    num_split_images = data_split_end - data_split_start

    if options.bootstrap:
        svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                      "modelb.%ssvm" % feature_str)
    else:
        svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                      "model1.%ssvm" % feature_str)
    
    lm = read_liblinear_model(svm_model_file)
    
    w = lm.w[:-1]
#    data_dim = float(w.size)
#    w_thresh = np.sort(np.abs(w))[int(data_dim * .95)]
#    w[np.abs(w) < w_thresh] = 0.0

    b = lm.w[-1] * lm.bias
    
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
    
    num_non_annotation_images = len(
                config_opts[0].non_annotation_image_inds[options.set_type])
    
    all_scores = []
    
    for i, image_ind in enumerate(
                config_opts[0].non_annotation_image_inds[
                    options.set_type][data_split_start:data_split_end]):
   
        print("{0}/{1}: {2} - {3}".format(
            i,
            num_split_images,
            image_ind,
            config_opts[0].image_sets_fp[options.set_type][image_ind]))
    
        for fs in feature_sources:
            fs.init_frame(image_ind)
            n_scales = len(fs.scales)
    
        for si in xrange(n_scales):
            for fs in feature_sources:
                locs, scale = fs.init_scale(si)
    
            num_y, num_x = locs[0].shape
            n_total = locs[0].size
    
            scale_scores = np.empty((num_y, num_x), dtype='float32')
    
            for yind in range(0, num_y, 10):
                ymin = yind
                ymax = min(num_y, yind + 10)
    
                feas = [fs.get_features_from_scale(
                    ymin=yind, ymax=ymax,
                    xmin=0, xmax=num_x) for fs in feature_sources]

                feas = np.concatenate(feas, axis=2)

                scale_scores[ymin:ymax, :] = (feas *
                    w[np.newaxis, np.newaxis, :]).sum(axis=2) + b
    
            num_fp = (scale_scores >= 0).sum()
    
            print("Im: {0}, Scale: {1:.2f}, FP: {2}".format(image_ind, scale, num_fp))
            for score in scale_scores.flat:
                all_scores.append(score)

    all_scores = np.array(all_scores)

    return all_scores

def classify_subrange_tup(args):
    return classify_subrange(*args)

if __name__ == '__main__':
    bootstrap_str = ''
    if options.bootstrap:
        bootstrap_str = 'b'

    target_file = os.path.join( 
        config_opts[0].result_path,
        "non_annotated_results_%s.%s%s.npy" %
            (options.set_type, feature_str, bootstrap_str))
   
    if options.data_split_ind != -1:
        all_scores = classify_subrange(options.data_split, options.data_split_ind)
    else:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(n_cpus)

        scores = pool.map(classify_subrange_tup, zip(
            [options.data_split for x in range(options.data_split)],
            range(options.data_split)))

        all_scores = np.hstack(scores)

    np.save(target_file, all_scores)
