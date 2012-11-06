import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.misc import imsave
import vivid

from config import ConfigOpt
from opt import options

from feature_source import *

config_sources = options.config_source.split(',')

config_opts = [ConfigOpt(cs) for cs in config_sources]

fv = vivid.ImageSource(imlist=config_opts[0].image_sets_fp[options.set_type])
cs = vivid.ConvertedSource(
    fv,
    target_type=vivid.cv.CV_32FC3,
    scale=1.0 / 255.0)
gv = vivid.GreySource(cs)
ss = vivid.ScaledSource(gv, 1.0)

feature_sources = []

for cos in config_opts:
    if cos.feature.fea_type == 'flex':
        print("Dictionary file: {0}".format(cos.dictionary_file))
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

num_annotations = len(config_opts[0].annotation_inds[options.set_type])

feature_dims = np.array([cos.feature.feature_dim for cos in config_opts])
feature_dims_cs = np.cumsum(feature_dims)

all_features = np.zeros((num_annotations, feature_dims.sum()), dtype='f')

for annotation_ind, annotation_image_ind in zip(
        config_opts[0].annotation_inds[options.set_type],
        config_opts[0].annotation_image_inds[options.set_type]):

    print("{0}: {1}").format(
        counter,
        config_opts[0].image_sets_fp[options.set_type][annotation_image_ind])

    annotation = config_opts[0].annotations[annotation_ind]

    max_size = np.max(annotation.size)
    max_dim = np.argmax(annotation.size)

    o_height = config_opts[0].object_size[0]
    o_width = config_opts[0].object_size[1]
    scaling = float(config_opts[0].object_size[max_dim]) / max_size

    annotation_location = [annotation.loc[0], annotation.loc[1], scaling]

    ss.scale = scaling

    gr_frame = vivid.cvmat2array(gv.get_frame(annotation_image_ind))

    scaled_frame = vivid.cvmat2array(ss.get_frame(annotation_image_ind))

    scaled_loc = np.array(annotation.loc) * scaling
    scaled_size = np.array(annotation.size) * scaling

    scaled_fixed_loc = (scaled_loc -
        (np.array(config_opts[0].object_size) - np.array(scaled_size)) / 2.0)

    scaled_fixed_loc = (
        (scaled_fixed_loc / config_opts[0].feature.window_stride).astype('int') *
        config_opts[0].feature.window_stride)

    margin_up = -scaled_fixed_loc[0]
    margin_left = -scaled_fixed_loc[1]
    margin_down = scaled_fixed_loc[0] + o_height - scaled_frame.shape[0] + 1
    margin_right = scaled_fixed_loc[1] + o_width - scaled_frame.shape[1] + 1
    
    low = 0

    for fea_i, fs in enumerate(feature_sources):
        fs.init_frame(
            annotation_image_ind, scales=[scaling],
            margin_left=margin_left, margin_up=margin_up,
            margin_right=margin_right, margin_down=margin_down)
    
        locs = fs.init_scale(0)
        feas = fs.get_features_from_scale()
    
        all_features[counter,low : feature_dims_cs[fea_i]] = feas[0,0]
        low += feature_dims_cs[fea_i]

    counter += 1

feature_str = ''
for cos in config_opts:
    feature_str += cos.feature_str

target_file = os.path.join(
    config_opts[0].feature_path,
    "annotation_features_%s.%snpy" % (options.set_type, feature_str))

assert(np.all((all_features * all_features).sum(axis=1) > 0.9999))

print "saving to %s"%target_file
np.save(target_file, all_features)
