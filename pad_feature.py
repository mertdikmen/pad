import numpy as np
import cPickle as pickle

import vivid

class PADFeatureSource:
    """ FeatureSource computes the block pad features """
    def __init__(self, origin, feature_model):
        self.fm = feature_model
        self.ff = vivid.FlexibleFilter(
                origin,
                self.fm.words,
                vivid.FF_OPTYPE_COSINE)

    def get_features(self, frame_no):
        assignments, weights = self.ff.filter_frame_cuda(frame_no)

        apron = self.fm.patch_size / 2

        cell_histograms = vivid.cell_histogram_dense_c(
                        assignments,
                        weights,
                        self.fm.dictionary_size,
                        self.fm.cell_size,
                        apron, apron, apron, apron)

        return cell_histograms

class PADFeatureModel:
    def __init__(self, words):
        self.words = words

        self.dictionary_size = words.shape[0]

        self.patch_size = words.shape[1] #in terms of pixels
        assert(words.shape[1] == words.shape[2])

        self.cell_size = 8 #in terms of patches (overlapping)

        self.block_size = 2 #in terms of cells (non-overlapping)

    def write(self, file_name):
        """Write the PAD feature model to disk"""
        fp = open(file_name, 'w+')
        pickle.dump(self, fp)
        fp.close()

class PADObjectModel:
    def __init__(self, words, svm_model, shape):
        self.fm = PADFeatureModel(words)
        self.svm = svm_model
        self.shape = shape
 
        self.svm_dm = None
    @classmethod
    def read(cls, file_name):
        model = pickle.load(open(file_name, 'r'))
        return cls(model.fm.words, model.svm, model.shape)

    def write(self, file_name):
        """Write the PAD object model to disk"""
        fp = open(file_name, 'w+')
        pickle.dump(self, fp)
        fp.close()

    def classify(self, block_feature_image):
        if (self.svm_dm == None):
            self.svm_dm = vivid.DeviceMatrix(
                    self.svm.w[:-1].reshape((-1, self.fm.dictionary_size)))
        
        retval = vivid.pwdot_cuda(block_feature_image, self.svm_dm)

        return retval
