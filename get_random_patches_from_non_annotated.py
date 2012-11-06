import vivid
import sys
import numpy as np
import os
if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle

from scipy.signal import convolve2d

from config import ConfigOpt
from opt import options

config_opt = ConfigOpt(options.config_source)

target_file = config_opt.patch_file

print("Will write to: {0}".format(target_file))

fv = vivid.ImageSource(imlist=config_opt.image_sets_fp[options.set_type])
cs = vivid.ConvertedSource(fv, target_type=vivid.cv.CV_32FC3, scale=1.0/255.0)
gv = vivid.GreySource(cs)

patches = np.zeros((
    config_opt.num_patches,
    config_opt.feature.cell.patch_size * config_opt.feature.cell.patch_size),
    dtype='float32')

patch_pixel_map_x,patch_pixel_map_y = np.meshgrid(
    np.arange(-(config_opt.feature.cell.patch_size/2),
               (config_opt.feature.cell.patch_size/2)+1),
    np.arange(-(config_opt.feature.cell.patch_size/2),
               (config_opt.feature.cell.patch_size/2)+1))

patch_margin = config_opt.feature.cell.patch_size / 2

num_images = len(config_opt.non_annotation_image_inds[options.set_type])

total_area = 0

for image_ind in config_opt.non_annotation_image_inds[options.set_type]:
    image = vivid.cvmat2array(gv.get_frame(image_ind))
    size = np.array(image.shape) - patch_margin * 2
    total_area += np.prod(size)

patch_count = 0
for i, image_ind in enumerate(
        config_opt.non_annotation_image_inds[options.set_type]):

    print("{0}/{2}: {1}".format(
        i,
        config_opt.image_sets_fp[options.set_type][image_ind],
        num_images))
    im = vivid.cvmat2array(gv.get_frame(image_ind))
    size = np.array(im.shape)

    height, width = size

    filt = np.ones((config_opt.feature.cell.patch_size,
                    config_opt.feature.cell.patch_size))

    num_patches_image = np.ceil(float(np.prod(size - patch_margin * 2)) / total_area *
                                config_opt.num_patches)

    if (patch_count + num_patches_image > config_opt.num_patches):
        num_patches_image = config_opt.num_patches - patch_count

    im_means = convolve2d(im, filt / (config_opt.feature.cell.patch_size *
                                      config_opt.feature.cell.patch_size),
                          mode='same')

    im_sq_means = convolve2d(im*im, filt, mode='same')

    patch_mags = (im_sq_means - im_means*im_means *
        (config_opt.feature.cell.patch_size *
         config_opt.feature.cell.patch_size))

    patch_mags[patch_mags < 0] = 0

    patch_mags = np.sqrt(patch_mags)

    patch_mags[:patch_margin, :] = 0
    patch_mags[-patch_margin:, :] = 0
    patch_mags[:, :patch_margin] = 0
    patch_mags[:, -patch_margin:] = 0

    patch_mags_cs = np.cumsum(patch_mags.flatten()).reshape((height,width))

    rand_vals = np.random.random(num_patches_image) * patch_mags.sum()
    rand_vals.sort()

    y_ind = np.zeros(num_patches_image, dtype='int')
    x_ind = np.zeros(num_patches_image, dtype='int')
    ri = 0
    for index, pmc in np.ndenumerate(patch_mags_cs):
        while(ri < num_patches_image and rand_vals[ri] < pmc):
            y_ind[ri] = index[0]
            x_ind[ri] = index[1]
            ri += 1

    assert(ri == num_patches_image)

    x_ind = x_ind[:,np.newaxis,np.newaxis] + patch_pixel_map_x[np.newaxis,:,:]
    y_ind = y_ind[:,np.newaxis,np.newaxis] + patch_pixel_map_y[np.newaxis,:,:]

    patches[patch_count:patch_count + num_patches_image] = (
        im[y_ind, x_ind].reshape((num_patches_image, -1)) )

    patch_count += num_patches_image

patch_means = np.mean(patches, axis=1)

patches = patches - patch_means[:,np.newaxis]

print("Writing to {0}".format(target_file))

np.save(target_file, patches)
