#!/usr/bin/python2
from object_annotation import ObjectAnnotation
import pickle
import numpy as np
import vivid
import os
from pad_feature import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Harvest features from annotated files")
parser.add_argument('annotation_file', type=str)
parser.add_argument('source_directory', type=str)
parser.add_argument('model_file', type=str)

options = parser.parse_args()

pom =  PADObjectModel.read(options.model_file)

apron = pom.fm.ff_apron()
window_size = np.array(pom.window_size())

annotations = pickle.load(open(options.annotation_file,'r'))

image_list = [os.path.join(options.source_directory, annotation.file_name)
        for annotation in annotations]

fv = vivid.ImageSource(imlist=image_list)
cs = vivid.ConvertedSource(
        fv,
        target_type = vivid.cv.CV_32FC3,
        scale = 1.0 / 255.0)
gv = vivid.GreySource(cs)

#rescale and crop if necessary
cr = vivid.CroppedSource(cs, 0, 0, -1, -1)
ss = vivid.ScaledSource(cr, 1.0)

fs = PADFeatureSource(ss, pom.fm)

for i, annotation in enumerate(annotations):
    offset = (np.array(annotation.bb_location) - apron)

    print offset

    cr.set_crop_region(offset[0], offset[1], -1, -1)

    if i % 100 == 0:
        print("{}".format(i))

    cell_features = fs.get_features(i)

    break
#    block_mags = 
