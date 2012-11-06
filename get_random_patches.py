import vivid
import sys
import numpy as np
import os
if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle

from config import ConfigOpt
from opt import options

config_opt = ConfigOpt(options.config_source)

target_file = config_opt.patch_file[:-3] + 'nonbias.npy'

print("Will write to: {0}".format(target_file))

try:
    os.makedirs(target_path)
except:
    pass

fv = vivid.ImageSource(imlist=config_opt.image_sets_fp[options.set_type])
cs = vivid.ConvertedSource(fv, target_type = vivid.cv.CV_32FC3, scale = 1.0/255.0)
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

total_object_area = 0.0

patch_count = 0

for image_ind,annotation_ind in zip(
        config_opt.annotation_image_inds[options.set_type],
        config_opt.annotation_inds[options.set_type]):

    annotation = config_opt.annotations[annotation_ind]

    a_width = annotation.size[1]
    a_height = annotation.size[0]

    total_object_area += a_width * a_height

for image_ind,annotation_ind in zip(
        config_opt.annotation_image_inds[options.set_type],
        config_opt.annotation_inds[options.set_type]):

    annotation = config_opt.annotations[annotation_ind]

    a_width = annotation.size[1]
    a_height = annotation.size[0]

    margin_y = annotation.loc[0]
    margin_x = annotation.loc[1]

    num_patches_ann = (np.ceil(float(config_opt.num_patches) *
                       a_width * a_height / total_object_area))

    if (patch_count + num_patches_ann > config_opt.num_patches):
        num_patches_ann = config_opt.num_patches - patch_count

    y_ind = np.random.randint(0,
                              a_height - config_opt.feature.cell.patch_size,
                              num_patches_ann) + margin_y
    x_ind = np.random.randint(0,
                              a_width - config_opt.feature.cell.patch_size,
                              num_patches_ann) + margin_x

    x_ind = x_ind[:,np.newaxis,np.newaxis] + patch_pixel_map_x[np.newaxis,:,:]
    y_ind = y_ind[:,np.newaxis,np.newaxis] + patch_pixel_map_y[np.newaxis,:,:]

    frame = vivid.cvmat2array(gv.get_frame(image_ind))

    patches[patch_count:patch_count+num_patches_ann] = (
        frame[y_ind, x_ind].reshape((num_patches_ann, -1)) )

    print "%s: %d"%(annotation.image_name, num_patches_ann)

    patch_count += num_patches_ann

patch_means = np.mean(patches, axis=1)

patches = patches - patch_means[:,np.newaxis]

np.save(target_file, patches)
