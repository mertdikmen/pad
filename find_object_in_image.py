import numpy as np
import pylab as plt
import os
import multiprocessing

import vivid

from libsvm_helper import *

from feature_source import *

from config import ConfigOpt
from opt import options

from scipy.misc import imsave

from image_annotator import draw_bounding_box


def detect_and_write(input_images, output_images):
    DETECTION_THRESHOLD = -.5

    config_opt = ConfigOpt(options.config_source)

    if options.bootstrap:
        svm_model_file = os.path.join(config_opt.svm_model_path, "modelb.svm")
    else:
        svm_model_file = os.path.join(config_opt.svm_model_path, "model1.svm")

    lm = read_liblinear_model(svm_model_file)

    w = lm.w[:-1]
    b = lm.w[-1] * lm.bias

    fv = vivid.ImageSource(imlist=input_images)
    cs = vivid.ConvertedSource(fv,
                               target_type=vivid.cv.CV_32FC3,
                               scale=1.0 / 255.0)
    gv = vivid.GreySource(cs)

    clusters = np.load(config_opt.dictionary_file).reshape((
        config_opt.feature.cell.dictionary_size,
        config_opt.feature.cell.patch_size,
        config_opt.feature.cell.patch_size))

    fs = FeatureSource(gv, config_opt.feature, clusters)

    all_scores = []

    for fi, input_image, output_image in zip(
        xrange(len(input_images)), input_images, output_images):

        print('Image: {0}'.format(input_image))

        fs.init_frame(fi)

        frame = vivid.cvmat2array(cs.get_frame(fi))[:, :, ::-1]

        while True:
            try:
                locs, scale = fs.init_next_scale()

                print("Processing scale: {0:.2f}\t".format(scale))

                num_y, num_x = locs[0].shape

                scale_scores = np.empty((num_y, num_x), dtype='float32')

                for yind in range(0, num_y, 10):
                    ymin = yind
                    ymax = min(num_y, yind + 10)

                    feas = fs.get_features_from_scale(
                        ymin=yind, ymax=ymax,
                        xmin=0, xmax=num_x)

                    scale_scores[ymin:ymax, :] = (feas *
                        w[np.newaxis, np.newaxis, :]).sum(axis=2) + b

                detections = scale_scores >= DETECTION_THRESHOLD

                for yi, xi, detection_score in zip(
                        locs[0][detections],
                        locs[1][detections],
                        scale_scores[detections]):

                    if detection_score >= 0:
                        box_color = [1.0, 0, 0]
                    else:
                        box_color = [1.0, 1.0, 0]

                    print("Detection at: y: {0}, x: {1}, s: {2:.2f}".format(
                            yi, xi, scale))

                    frame = draw_bounding_box(frame, np.array([yi, xi, scale]),
                                              text="%.3f" % detection_score,
                                              color=box_color)
            except EndOfScales:
                break

        imsave(output_image, frame)


def detect_and_write_tup(args):
    return detect_and_write(*args)

if __name__ == '__main__':
    if options.source_dir == None:
        scaled_sources = detect_and_write([options.input_image],
                                          [options.output_image])
    else:
        import imghdr
        file_list = os.listdir(options.source_dir)

        out_dir = os.path.join(options.source_dir, 'out')

        image_list = []
        output_list = []
        for f in file_list:
            full_path = os.path.join(options.source_dir, f)
            try:
                image_type = imghdr.what(full_path)
                if image_type != None:
                    image_list.append(full_path)
                    output_list.append(os.path.join(out_dir, f))
            except IOError:
                pass
        if len(image_list) > 0:
            try:
                os.makedirs(out_dir)
            except:
                os.system('rm {0}/*'.format(out_dir))

            n_cpus = multiprocessing.cpu_count()
            list_parts = np.unique(
                np.linspace(0, len(image_list), n_cpus + 1).astype('int'))

            image_list_part = [
                image_list[list_parts[i]:list_parts[i + 1]]
                    for i in range(len(list_parts) - 1)]

            output_list_part = [
                output_list[list_parts[i]:list_parts[i + 1]]
                    for i in range(len(list_parts) - 1)]

            pool = multiprocessing.Pool(n_cpus)

            pool.map(detect_and_write_tup, zip(image_list_part,
                                               output_list_part))
