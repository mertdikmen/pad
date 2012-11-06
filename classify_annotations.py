import numpy as np
import os
import pylab as plt
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

if options.bootstrap:
    svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                  "modelb.%ssvm" % feature_str)
else:
    svm_model_file = os.path.join(config_opts[0].svm_model_path,
                                  "model1.%ssvm" % feature_str)

lm = read_liblinear_model(svm_model_file)

w = lm.w[:-1]
#data_dim = float(w.size)
#w_thresh = np.sort(np.abs(w))[int(data_dim * .95)]
##plt.ion()
##plt.plot(np.sort(np.abs(w)))
##raw_input()
#w[np.abs(w) < w_thresh] = 0.0

b = lm.w[-1] * lm.bias

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

all_scores = np.zeros(num_annotations)

for annotation_ind, annotation_image_ind in zip(
        config_opts[0].annotation_inds[options.set_type], 
        config_opts[0].annotation_image_inds[options.set_type]):

    print "%d: %s" % (
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

    margin_up = -int(annotation.loc[0] * scaling)
    margin_left = -int(annotation.loc[1] * scaling)
    margin_down = -int(scaled_frame.shape[0] -
        (annotation.loc[0] * scaling + config_opts[0].object_size[0]))
    margin_right = -int(scaled_frame.shape[1] -
        (annotation.loc[1] * scaling + config_opts[0].object_size[1]))
#    print margin_right

    for fs in feature_sources:
        fs.init_frame(
            annotation_image_ind, scales=[scaling],
            margin_left=margin_left, margin_up=margin_up,
            margin_right=margin_right, margin_down=margin_down)
    
        locs = fs.init_scale(0)

    feas = [fs.get_features_from_scale() for fs in feature_sources]

    feas = np.concatenate(feas, axis=2)

    feas = feas[0][0, :].reshape((1,-1))

    all_scores[counter] = np.dot(feas,w) + b 

    counter += 1 

bootstrap_str = ''
if options.bootstrap:
    bootstrap_str = 'b.'

target_file = os.path.join(
    config_opts[0].result_path, 
    "annotated_results_%s.%s%snpy" %
        (options.set_type, feature_str, bootstrap_str))

print "saving to %s"%target_file
np.save(target_file, all_scores)
