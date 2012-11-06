import numpy as np
import sys
import os

PY_LIBLINEAR_PATH = '/home/mert/Research/py_liblinear'

sys.path.append(PY_LIBLINEAR_PATH)

import liblinear_python

from config import ConfigOpt
from opt import options

config_sources = options.config_source.split(',')

config_opts = [ConfigOpt(cs) for cs in config_sources]

feature_str = ''
for cos in config_opts:
    feature_str += cos.feature_str

if options.bootstrap:
    target_file = os.path.join(config_opts[0].svm_model_path,
                               "modelb.%ssvm" % feature_str)
else:
    target_file = os.path.join(config_opts[0].svm_model_path,
                               "model1.%ssvm" % feature_str)

print "Will write model to %s"%target_file

solver_list = {'L2R_LR': 0, 'L2R_L2LOSS_SVC_DUAL': 1, 'L2R_L2LOSS_SVC': 2,
               'L2R_L1LOSS_SVC_DUAL': 3, 'MCSVM_CS': 4, 'L1R_L2LOSS_SVC': 5,
               'L1R_LR': 6, 'L2R_LR_DUAL': 7}

solver_type = 'L2R_L1LOSS_SVC_DUAL'

normalize_flag = 0

pos_source_file = os.path.join(config_opts[0].feature_path,
                "annotation_features_training.%snpy" % feature_str)

print "Loading %s"%pos_source_file
data_mat = np.load(pos_source_file).astype('float32')

if options.tfidf == 1:
    data_mat = data_mat * tfidf_w[np.newaxis,:]

ll = liblinear_python.LibLinearPy()
ll.set_solver(solver_list[solver_type])
ll.add_data(data_mat, normalize_flag, 1)

for i in range(options.data_split):
    neg_source_file = os.path.join(
        config_opts[0].feature_path,
        "non_annotation_features_%s.%s%s%03d:%03d.npy" % (
            options.set_type,
            feature_str,
            options.bootstrap_str,
            i,
            options.data_split))
 
    print "Loading %s"%neg_source_file
    data_mat = np.load(neg_source_file).astype('f')
        
    n_data = data_mat.shape[0]

    ll.add_data(data_mat, normalize_flag, -1)

ll.set_weights(1.0, 1.0)

ll.train_svm(target_file,1)
