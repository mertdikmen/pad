import numpy as np
import cPickle as pickle
import scipy.signal
import vivid

class PADFeatureSource:
    """ FeatureSource computes the block pad features
    origin: a vivid source (currently supported GreySource with CV_32FC1)
    """
    def __init__(self, origin, feature_model):
        self.fm = feature_model
        self.ff = vivid.FlexibleFilter(
                origin,
                self.fm.words,
                vivid.FF_OPTYPE_COSINE)

    def get_features(self, frame_no):
        """
        Computes the cell histograms of PAD words
        """
        assignments, weights = self.ff.filter_frame_cuda(frame_no)

        apron = self.fm.ff_apron()

        cell_histograms = vivid.cell_histogram_dense_c(
                        assignments,
                        weights,
                        self.fm.dictionary_size,
                        self.fm.cell_size,
                        apron, apron, apron, apron)

        cell_mags = np.sum(cell_histograms, axis=2)

        cell_weights = scipy.signal.correlate2d(
                vivid.rectangle_sum(cell_mags, 2, 2), np.ones((2, 2)))

        #take care of the edge and corner cases
        cell_weights[:,0] *= 2
        cell_weights[:,-1] *= 2
        cell_weights[0,:] *= 2
        cell_weights[-1,:] *= 2

        cell_weights[0,0] *= 2 
        cell_weights[-1,0] *= 2 
        cell_weights[0,-1] *= 2 
        cell_weights[-1,-1] *= 2 

        return cell_histograms / cell_weights[:,:,np.newaxis]

class PADFeatureModel:
    def __init__(self, words):
        self.words = words

        self.dictionary_size = words.shape[0]

        self.patch_size = words.shape[1] #in terms of pixels
        assert(words.shape[1] == words.shape[2])

        self.cell_size = 8 #in terms of overlapping patches

        self.block_size = 2 #in terms of non-overlapping cells

    def ff_apron(self):
        return self.patch_size / 2

    def write(self, file_name):
        """Write the PAD feature model to disk"""
        fp = open(file_name, 'w+')
        pickle.dump(self, fp)
        fp.close()

class PADObjectModel:
    def __init__(self, words, svm_model, shape):
        self.fm = PADFeatureModel(words)
        self.svm = svm_model
        self.shape = shape #in terms of overlapping blocks
 
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
        """Convolve the classifier with the PAD cell histogram"""
        if (self.svm_dm == None):
            self.svm_dm = vivid.DeviceMatrix(
                    self.svm.w[:-1].reshape((-1, self.fm.dictionary_size)))
        
        retval = vivid.pwdot_cuda(block_feature_image, self.svm_dm)

        return retval

    def subwindow_feature(cell_histograms, cell_mags, cell_ind_y, cell_ind_x):
        subwindow_cm = cell_mags[
                cell_ind_y : cell_ind_y + self.shape[0],
                cell_ind_x : cell_ind_x + self.shape[1]]

        subwindow_ch = cell_histograms[
                cell_ind_y : cell_ind_y + self.shape[0],
                cell_ind_x : cell_ind_x + self.shape[1], :]

    def window_size(self):
        block_h = self.fm.cell_size * self.fm.block_size * self.shape[0]
        block_w = self.fm.cell_size * self.fm.block_size * self.shape[1]

        return (block_h + self.fm.cell_size, block_w + self.fm.cell_size)
