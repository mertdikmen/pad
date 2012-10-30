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

annotations = pickle.load(open(options.annotation_file,'r'))

image_list = [os.path.join(options.source_directory, annotation.file_name)
        for annotation in annotations]

fv = vivid.ImageSource(imlist=image_list)
cs = vivid.ConvertedSource(
        fv,
        target_type = vivid.cv.CV_32FC3,
        scale = 1.0 / 255.0)
gv = vivid.GreySource(cs)
fs = PADFeatureSource(gv, pom.fm)

for i in range(len(image_list)):
    cell = fs.get_features(i)
