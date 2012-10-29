import numpy as np
from pad_feature import PADObjectModel
import libsvm_helper as lh

lm = lh.read_liblinear_model('models/modelb.svm')

w = lm.w[:-1]
b = lm.w[-1] * lm.bias

dictionary = np.load('seed.non_object_sp_kmeans_d.0100.biased.npy')
dictionary = dictionary.reshape((-1,3,3))

object_shape = [15,7]

pom = PADObjectModel(dictionary, lm, object_shape)

pom.write("example_pom.pkl")
