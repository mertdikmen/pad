import vivid
import numpy as np
from scipy import signal

from config import FeatureOpt

from dfl_exceptions import *

PRINT_PROFILE = False

import time

EPSILON = 1e-6

def get_lbp_assignments(lbpsource, frame_num):
    assign = lbpsource.get_lbp(frame_num).astype('float32')
    weights = np.ones(assign.shape, dtype='float32')
    return assign, weights

def get_flex_assignments(flexfilter, frame_num):
    res = flexfilter.filter_frame_cuda(frame_num)

    return res[0], res[1]

def get_uniform_assigmnents(uniformfilter, frame_num):
    weights = uniformfilter.get_frame(frame_num)
    assign = np.zeros(weights.shape, dtype='float32')
    return assign, weights

class FeatureSource(object):
    """ Top level feature source
    This will take care of all computation of window locations and rescaling
    The logic is to separate actual feature computation and figuring out
    where to compute
    """
    def __init__(self, origin, feature_opt, clusters=None):
        self.origin = vivid.CachedSource(origin, cache_size=1)
        self.feature_opt = feature_opt

        if self.feature_opt.fea_type == 'flex':
            self.clusters = clusters

            ff_optype = {'kmeans': vivid.FF_OPTYPE_EUCLIDEAN,
                         'sp_kmeans': vivid.FF_OPTYPE_COSINE}

            ff_optype = {'abs_cosine': vivid.FF_OPTYPE_COSINE}

            self.assignment_source = vivid.FlexibleFilter(
                        self.origin, self.clusters,
                        ff_optype[feature_opt.cell.distance_type])
            self.get_assignments = get_flex_assignments

        elif self.feature_opt.fea_type == 'lbp':
            self.assignment_source = vivid.LocalBinaryPatternSource(
                        self.origin)
            self.get_assignments = get_lbp_assignments

    def _init_scale(self, scale_ind, dbg=False):
        """
        frame_size is the real image size

        frame_size + (margins in each direction) is the zero padded image size

        We calculate the cell histograms only on the cells that are fully
        inside the real image.  Otherwise the cell histograms are considered
        all zeros.

        First we need to figure out the size & the boundaries of the cell grid
         on the padded image and the real image.

        """
        if scale_ind >= len(self.scales):
            raise EndOfScales

        frame_size = [self.scaled_sources[scale_ind].get_frame(
                        self.feature_frame).height,
                      self.scaled_sources[scale_ind].get_frame(
                        self.feature_frame).width]

        #self.ff.origin = self.scaled_sources[scale_ind]
        self.assignment_source.origin = self.scaled_sources[scale_ind]

        # This is the entire cell grid.
        # The ranges can be outside the image boundaries to account for
        # the cases when the window is partially outside the image
        x_start = -self.margins[2]
        x_stop = (frame_size[1] + self.margins[3] -
                  self.feature_opt.cell.cell_size + 1)

        y_start = -self.margins[0]
        y_stop = (frame_size[0] + self.margins[1] -
                  self.feature_opt.cell.cell_size + 1)
        cell_grid = np.meshgrid(np.arange(x_start, x_stop, 
                                          self.feature_opt.cell.cell_size),
                                np.arange(y_start, y_stop, 
                                          self.feature_opt.cell.cell_size))

#        import pdb;pdb.set_trace()

        cell_grid_start_x = -self.margins[2]
        cell_grid_start_y = -self.margins[0]
        cell_grid_stop_x = np.max(cell_grid[0][0])
        cell_grid_stop_y = np.max(cell_grid[1][:, 0])

        num_cells_x = cell_grid[1].shape[1]
        num_cells_y = cell_grid[0].shape[0]

        # The cells that fall within the image and can be populated with
        # histograms
        min_offset = self.feature_opt.cell.patch_size / 2

        cell_grid_start_x_image = np.min(
            cell_grid[0][cell_grid[0] >= min_offset])
        cell_grid_start_y_image = np.min(
            cell_grid[1][cell_grid[1] >= min_offset])

        cell_grid_stop_x_image = np.max(cell_grid[0][
             cell_grid[0] < frame_size[1] -
             self.feature_opt.cell.cell_size])

        cell_grid_stop_y_image = np.max(cell_grid[1][
             cell_grid[1] < frame_size[0] -
             self.feature_opt.cell.cell_size])

        # Indices of the image cells in the cell grid
        cell_ind_start_y = ((cell_grid_start_y_image - cell_grid_start_y) /
            self.feature_opt.cell.cell_size)

        cell_ind_start_x = ((cell_grid_start_x_image - cell_grid_start_x) /
            self.feature_opt.cell.cell_size)

        cell_ind_stop_y = ((cell_grid_stop_y_image - cell_grid_start_y) /
            self.feature_opt.cell.cell_size) + 1  # non inclusive range

        cell_ind_stop_x = ((cell_grid_stop_x_image - cell_grid_start_x) /
            self.feature_opt.cell.cell_size) + 1  # non inclusive range

        #patch dictionary assignments
        tic = time.time()
        assignments, weights = self.get_assignments(self.assignment_source,
                                               self.feature_frame)

        if PRINT_PROFILE:
            print "Filtering time %f" % (time.time() - tic)

        #cell histograms
        tic = time.time()

        cell_hist = np.zeros((num_cells_y, num_cells_x,
                              self.feature_opt.cell.dictionary_size),
                              dtype='float32')

        cell_hist[cell_ind_start_y:cell_ind_stop_y,
                  cell_ind_start_x:cell_ind_stop_x,
                  :] = vivid.cell_histogram_dense(
                        assignments,  # assignments
                        weights,  # similarities
                        self.feature_opt.cell.dictionary_size,
                        self.feature_opt.cell.cell_size,
                        [cell_grid_start_y_image, cell_grid_start_x_image],
                        [cell_grid_stop_y_image +
                            self.feature_opt.cell.cell_size,
                         cell_grid_stop_x_image +
                            self.feature_opt.cell.cell_size])

        if self.feature_opt.fea_type == 'lbp':
            cell_hist[:,:,0] = 0

        if PRINT_PROFILE:
            print "Cell histogram time %f" % (time.time() - tic)

        #block histograms
        tic = time.time()
        if self.feature_opt.block.cell_grouping == 'concatenate':
            #block_mags = vivid.group_cell_histograms(
            #    np.abs(cell_hist),
            #    self.feature_opt.block.block_size,
            #    self.feature_opt.block.block_size,
            #    1)
            #
            #block_mags = np.sqrt(block_mags)
            #
            #n_blocks = block_mags.shape
            #
            #cell_hist_temp = np.zeros_like(cell_hist)
            #
            #import pdb; pdb.set_trace()

            #for by in range(self.feature_opt.block.block_size):
            #    for bx in range(self.feature_opt.block.block_size):
            #        cell_hist_temp[by:by+n_blocks[0], bx : bx + n_blocks[1]] += (
            #            cell_hist[by:by+n_blocks[0], bx : bx + n_blocks[1]] /
            #            block_mags )
                        
            self.block_hist = vivid.group_cell_histograms(
                            cell_hist,
                            self.feature_opt.block.block_size,
                            self.feature_opt.block.block_size,
                            1)

        elif self.feature_opt.block.cell_grouping == 'add':
            self.block_hist = vivid.add_cell_histograms(
                            cell_hist,
                            self.feature_opt.block.block_size,
                            self.feature_opt.block.block_size,
                            1)

        if self.feature_opt.block.block_normalization == 'l2':
            cell_mags = np.reshape((cell_hist * cell_hist).sum(axis=2),
                                   (cell_hist.shape[0], cell_hist.shape[1], 1))
            block_mags = vivid.group_cell_histograms(
                            cell_mags,
                            self.feature_opt.block.block_size,
                            self.feature_opt.block.block_size, 1)

            block_mags = np.sqrt(block_mags.sum(axis=2)) # + EPSILON
            block_mags[block_mags == 0] = 1.0
            self.block_mags = block_mags

            self.block_hist = self.block_hist / block_mags[:, :, np.newaxis]

        elif self.feature_opt.block.block_normalization == 'l1sqrt':
            block_mags = vivid.group_cell_histograms(
                            np.abs(cell_hist),
                            self.feature_opt.block.block_size,
                            self.feature_opt.block.block_size,
                            1)

            block_mags = block_mags.sum(axis=2)
            block_mags[block_mags == 0] = 1.0
            self.block_hist = self.block_hist / block_mags[:, :, np.newaxis]
            self.block_hist = np.sqrt(self.block_hist)

        if PRINT_PROFILE:
            print "Block histogram time %f" % (time.time() - tic)

        window_grid = [cell_grid[1][:-self.feature_opt.blocks_per_win[0],
                                    :-self.feature_opt.blocks_per_win[1]],
                       cell_grid[0][:-self.feature_opt.blocks_per_win[0],
                                    :-self.feature_opt.blocks_per_win[1]]]

        #chmag = np.sqrt(np.sum(self.ch * self.ch, axis=2))
        #print("C: {0}, {1}".format(chmag.min(),chmag.max()))
        #print("B: {0}, {1}".format(self.block_mags.min(),self.block_mags.max()))

        return window_grid

    def get_features_from_scale(self,
                     ymin=None, ymax=None,
                     xmin=None, xmax=None):
        tic = time.time()
        
        if ymin == None or xmin == None or ymax == None or xmax == None:
            feature_grid = vivid.group_cell_histograms(
               self.block_hist,
               int(self.feature_opt.blocks_per_win[0]),
               int(self.feature_opt.blocks_per_win[1]),
               1)
        else:
            feature_grid = vivid.group_cell_histograms(
                self.block_hist[
                    ymin:(ymax - 1) + int(self.feature_opt.blocks_per_win[0]),
                    xmin:(xmax - 1) + int(self.feature_opt.blocks_per_win[1]),
                    :],
                int(self.feature_opt.blocks_per_win[0]),
                int(self.feature_opt.blocks_per_win[1]),
                1)

        if self.feature_opt.feature_normalization:
            # if we normalized the blocks, we know the norm already
            #if self.feature_opt.block.block_normalization == 'l2':
            #    fea_sum = np.sum(feature_grid * feature_grid,axis=2)
            #   
            #    feature_mag = np.sqrt(
            #        self.feature_opt.blocks_per_win[0] *
            #        self.feature_opt.blocks_per_win[1])
            #    feature_grid /= feature_mag
            #else:
            feature_mag = (
                np.sqrt(np.sum(feature_grid * feature_grid, axis=2)) +
                EPSILON)
            
            #feature_sums = np.sqrt( (feature_grid * feature_grid).sum(axis=2)
                # + EPSILON)
            feature_grid = feature_grid / feature_mag[:, :, np.newaxis]
        # feature_sums[:,:,np.newaxis]

        #print "Feature compile time %f" % (time.time() - tic)

        #print "feature size: %d MB" % (feature_grid.nbytes / 1024 / 1024)
        #print feature_grid.shape

        return feature_grid

    def init_frame(self, frame_num,
                     margin_up=None, margin_down=None,
                     margin_left=None, margin_right=None,
                     scales=None):

        self.scales = scales

        self.feature_frame = frame_num

        self.last_scale = -1

        """
        this is the entry point, call this function to  compute features
        margin is in terms of window strides
        """
        min_offset = self.feature_opt.cell.patch_size / 2

        if margin_up == None:
            margin_up = -min_offset
        if margin_down == None:
            margin_down = -min_offset
        if margin_left == None:
            margin_left = -min_offset
        if margin_right == None:
            margin_right = -min_offset

        self.margins = [margin_up, margin_down, margin_left, margin_right]

        frame_size = np.array([self.origin.get_frame(frame_num).height,
                               self.origin.get_frame(frame_num).width])

        self.scaled_sources = []
        if self.scales == None:
            scale_factor = 1.0
            self.scales = []

            while (True):
                cs = vivid.CachedSource(vivid.ScaledSource(
                                           self.origin,
                                           scale=1.0 / scale_factor),
                                        cache_size=1)

                frame_size = np.array(vivid.cvmat2array(
                    cs.get_frame(self.feature_frame)).shape, dtype='float32')

                #see if we can fit a window in this scale
                h_windows = ((frame_size[0] - margin_up + margin_down) /
                    self.feature_opt.window_size[0])

                w_windows = ((frame_size[1] + margin_left + margin_right) /
                    self.feature_opt.window_size[1])

                #print "h: %.2f\t%.2f" % (h_windows, w_windows)
                if h_windows < 1 or w_windows < 1:
                    break

                self.scales.append(1.0 / scale_factor)
                scale_factor *= self.feature_opt.scale_step
                self.scaled_sources.append(cs)

        else:
            for s in scales:
                self.scaled_sources.append(
                    vivid.ScaledSource(self.origin, scale=s))

    def init_scale(self, scale_ind, dbg=False):
        loc = self._init_scale(scale_ind, dbg)
        return loc, self.scales[scale_ind]

    def init_next_scale(self):
        cur_scale_ind = self.last_scale + 1

        if cur_scale_ind >= len(self.scales):
            raise EndOfScales(len(self.scales))

        loc = self._init_scale(cur_scale_ind)

        self.last_scale += 1

        return loc, self.scales[cur_scale_ind]

#    def get_all_features(self):
#        features = []
#        locations = []
#
#        for si,ss in enumerate(self.scaled_sources):
#            fea, loc = self._feature_computation(si)
#            #fea, loc = self._feature_computation_test(ss, frame_num, yx_offset)
#            features.append(fea)
#            locations.append(loc)
#
#        return features, locations, self.scales
#
#
#    def _feature_computation_test(self, scaled_source, frame_num, yx_offset):
#        frame_size = [scaled_source.get_frame(frame_num).height,
#                      scaled_source.get_frame(frame_num).width]
#        num_clusters = self.clusters.shape[0]
#
#        similarity_threshold_scaler = 0.75
#
#        grid_locs = np.meshgrid(
#                        np.arange(
#                            yx_offset[1],
#                            (frame_size[1] - yx_offset[1] -
#                             self.feature_opt.window_size[1] + 1),
#                            self.feature_opt.cell.cell_size),
#                        np.arange(
#                            yx_offset[0],
#                            (frame_size[0] - yx_offset[0] -
#                             self.feature_opt.window_size[0] + 1),
#                            self.feature_opt.cell.cell_size)
#                        )
#
#        squared_source = vivid.SquaredSource(scaled_source)
#        average_filter = np.ones((1,
#                                  self.clusters.shape[1],
#                                  self.clusters.shape[2]),
#                                 dtype='float32')
#
#        tic = time.time()
#        mf = vivid.MultiFilterSource(scaled_source, None, self.clusters)
#        mf_square = vivid.MultiFilterSource(squared_source,
#                                            None,
#                                            average_filter)
#
#        mf_average = vivid.MultiFilterSource(scaled_source,
#                                             None,
#                                             (average_filter /
#                                              self.clusters.shape[1] /
#                                              self.clusters.shape[2])
#                                            )
#
#        fb_response = np.abs(mf.get_frame(frame_num))
#        squared_average = mf_square.get_frame(frame_num)
#        image_average = mf_average.get_frame(frame_num)
#
#        fb_sim = np.max(fb_response, axis=2)
#        toc = time.time()
#        print "\nFiltering time: %.2f" % (toc - tic)
#
#        similarity_thresholds = (squared_average -
#                                 image_average * image_average *
#                                 (self.clusters.shape[1] *
#                                  self.clusters.shape[2])
#                                )
#
#        similarity_thresholds = (np.sqrt(
#                                    np.maximum(0, similarity_thresholds)) *
#                                 similarity_threshold_scaler)
#
#        cluster_assignments = (
#            fb_response >= similarity_thresholds).astype('float32')
#
#        print("Float cast: %.2f" % (toc - tic))
#
#        tic = time.time()
#
#        ##python verification##
#        weight_vals = fb_response * cluster_assignments
#
#        im_height = fb_response.shape[0]
#        im_width = fb_response.shape[1]
#        num_bins = self.feature_opt.cell.dictionary_size
#
#        cell_hist = np.zeros((
#            (im_height - 2 * yx_offset[0]) / self.feature_opt.cell.cell_size,
#            (im_width - 2 * yx_offset[1]) / self.feature_opt.cell.cell_size,
#            num_bins), dtype='float32')
#
#        cell_size = self.feature_opt.cell.cell_size
#
#        start_y = int(yx_offset[0])
#        stop_y = int(im_height - yx_offset[0] - cell_size + 1)
#
#        start_x = int(yx_offset[1])
#        stop_x = int(im_width - yx_offset[1] - cell_size + 1)
#
#        for i, yi in enumerate(range(start_y, stop_y, cell_size)):
#            for j, xi in enumerate(range(start_x, stop_x, cell_size)):
#                cell_hist[i, j, :] = (
#                    weight_vals[yi:yi + cell_size, xi:xi + cell_size, :]
#                    ).sum(axis=0).sum(axis=0)
#
##        cell_hist = vivid.cell_histogram_dense_non_exclusive_binning(
##            cluster_assignments,
##            fb_response,
##            self.feature_opt.cell.cell_size,
##            yx_offset)
##
#        toc = time.time()
#        print "Cell histogramming: %.2f" % (toc - tic)
#
#        tic = time.time()
#        cell_mags = np.reshape((cell_hist * cell_hist).sum(axis=2),
#                               (cell_hist.shape[0], cell_hist.shape[1], 1))
#
#        block_mags = vivid.group_cell_histograms(
#                        cell_mags,
#                        self.feature_opt.block.block_size,
#                        self.feature_opt.block.block_size,
#                        1)
#
#        block_mags = np.sqrt(block_mags.sum(axis=2)) + EPSILON
#
#        block_hist = vivid.group_cell_histograms(
#                        cell_hist,
#                        self.feature_opt.block.block_size,
#                        self.feature_opt.block.block_size,
#                        1)
#
#        if self.feature_opt.block.block_normalization == 'l2':
#            #block_mags = np.sqrt( (block_hist*block_hist).sum(axis=2) )
#            #           + EPSILON
#
#            block_hist = block_hist / block_mags[:, :, np.newaxis]
#
#        feature_mag = np.sqrt(np.prod(self.num_blocks))
#
#        feature_grid = vivid.group_cell_histograms(
#                            block_hist,
#                            int(self.feature_opt.blocks_per_win[0]),
#                            int(self.feature_opt.blocks_per_win[1]),
#                            1)
#
#        if self.feature_opt.feature_normalization:
#            #feature_sums = np.sqrt( (feature_grid * feature_grid).sum(axis=2)
#            # + EPSILON)
#
#            feature_grid = feature_grid / feature_mag
#            # feature_sums[:,:,np.newaxis]
#
#        toc = time.time()
#        print "Other time: %.2f" % (toc - tic)
#
#        return feature_grid, grid_locs
#
#
#EPSILON = 1e-7
#
#def convolve_with_filter_bank(im, filter_b):
#    filter_bank_size = len(filter_b)
#    out_im = np.zeros((im.shape[0], im.shape[1], filter_bank_size),dtype='f')
#
#    for i,f in enumerate(filter_b):
#        out_im[:,:,i] = signal.correlate(im,f, mode='same')
#
#    return out_im
#
#""" Computes the cell locations based on the cell_size and the window_size
#Cells are disjoint (no overlap) and dense on a grid
#"""
#def _cell_locations(cell_size,window_size):
#    cell_offset_y = np.arange(0,window_size[0]-cell_size+1, cell_size)
#    cell_offset_x = np.arange(0,window_size[1]-cell_size+1, cell_size)
#
#    cell_locs_x, cell_locs_y = np.meshgrid(cell_offset_x, cell_offset_y)
#
#    cell_locs_x = cell_locs_x.ravel()
#    cell_locs_y = cell_locs_y.ravel()
#
#    return np.vstack((cell_locs_y, cell_locs_x)).T
#
#""" Given an image size compute the locations of the sliding windows where the classifier
#will be evaluated
#"""
#def _get_window_grid(image_size, window_size, filter_apron,  offset=0, stride=8):
#    processing_margin = filter_apron + offset  #+1 for gradient filter
#    start = processing_margin
#    end = image_size - window_size - processing_margin
#    
#    locs_y = np.arange(start,end[0]+1,step=stride)
#    locs_x = np.arange(start,end[1]+1,step=stride)
#    
#    if len(locs_y)>0 and len(locs_x) > 0:
#        return (locs_y, locs_x)
#    else:
#        return []
#
#""" If block normalization is selected this will compute the all groups of block_dimxblock_dim
#cell blocks
#"""
#def _cell_blocks(cell_size, window_size, block_dim, block_step = 1):
#    cy = np.arange(0,window_size[0]-cell_size+1, cell_size)
#    cx = np.arange(0,window_size[1]-cell_size+1, cell_size)
#
#    cell_block_map_x = np.arange(block_dim)
#    cell_block_map_y = np.arange(0,block_dim*len(cx),len(cx))
#    
#    cell_block_map = (cell_block_map_x[:,np.newaxis] + cell_block_map_y[np.newaxis,:]).ravel()
#
#    by = np.arange(0, len(cy)-block_dim+1, block_step)
#    bx = np.arange(0, len(cx)-block_dim+1, block_step)
#
#    blocks_x, blocks_y = np.meshgrid(bx,by)
#
#    blocks = (blocks_y * len(cx) + blocks_x).ravel()
#
#    block_cell_map = np.meshgrid(np.arange(block_dim), np.arange(0,len(cx)*block_dim,len(cx)))
#    block_cell_map = (block_cell_map[0] + block_cell_map[1]).ravel()
#
#    blocks = blocks[:,np.newaxis] + block_cell_map[np.newaxis,:]
#
#    return blocks.astype('int')
#
#
#""" If weight option is selected this will compute the patch weights centered around each valid pixel
#"""
#def _compute_patch_weight_mat(gradient_fr, patch_size):
#    base_weight_mat = np.sum((gradient_fr*gradient_fr), axis=2)
#    out_mat = np.zeros((gradient_fr.shape[0],gradient_fr.shape[1]), dtype='float32')
#    
#    integral_im = np.zeros((base_weight_mat.shape[0] +1 , base_weight_mat.shape[1] + 1), dtype='float32')
#
#    integral_im[1:,1:] = np.cumsum(np.cumsum(base_weight_mat,axis=0),axis=1)
#
#    height = base_weight_mat.shape[0]
#    width  = base_weight_mat.shape[1]
#
#    radius_lo = patch_size / 2
#    radius_hi = patch_size / 2
#
#    for i in range(radius_lo, height-radius_hi):
#        for j in range(radius_lo, width-radius_hi):
#            out_mat[i,j] = (integral_im[i-radius_lo,j-radius_lo] 
#                           +integral_im[i+radius_hi+1,j+radius_hi+1] 
#                           -integral_im[i-radius_lo,j+radius_hi+1] 
#                           -integral_im[i+radius_hi+1,j-radius_lo])
#
#    #sometimes the zero value can be tiny negative
#    out_mat[out_mat < 0] = 0
#
#    return np.sqrt(out_mat)
#
#def _compute_mean_sub_weight_mat(grey_fr, patch_size):
#    integral_im = np.zeros((grey_fr.shape[0] + 1, grey_fr.shape[1] + 1), dtype='float32')
#    integral_im[1:,1:] = np.cumsum(np.cumsum(grey_fr / (patch_size*patch_size),axis=0),axis=1) 
#    patch_means = integral_im[:-patch_size,:-patch_size] + integral_im[patch_size:,patch_size:] - integral_im[patch_size:,:-patch_size] - integral_im[:-patch_size,patch_size:]
#
#    integral_sq_im = np.zeros((grey_fr.shape[0] +1 , grey_fr.shape[1] + 1), dtype='float64')
#    integral_sq_im[1:,1:] = np.cumsum(np.cumsum(grey_fr*grey_fr,axis=0),axis=1)
#    patch_sq_sums = integral_sq_im[:-patch_size,:-patch_size] + integral_sq_im[patch_size:,patch_size:] - integral_sq_im[patch_size:,:-patch_size] - integral_sq_im[:-patch_size,patch_size:]
#
#    patch_radius = patch_size / 2
#
#    mean_sub_mags = np.zeros_like(grey_fr)
#
#    mean_sub_mags[patch_radius:-patch_radius,patch_radius:-patch_radius] = patch_sq_sums + (-patch_size*patch_size) * patch_means * patch_means
#
#    mean_sub_mags[mean_sub_mags < 0] = 0
#
#    #return mean_sub_mags
#    return mean_sub_mags
#
#""" This is the feature source
#"""
#class FlexibleFeatureSource:
#    def __init__(self, origin, clusters, dict_size, clustering_method,
#                 weighted_hist, patch_size, cell_size, window_size, block_size = 2,
#                 block_normalization=2, feature_normalization='norm',feature_pca=[],pca_dim=0):
#
#        self.origin = origin
#
#        self.window_size = window_size
#        self.dict_size = dict_size
#        self.patch_size = patch_size
#        self.cell_size = cell_size
#
#        self.block_normalization = block_normalization
#        self.feature_normalization = feature_normalization
#
#        if self.block_normalization == 0:
#            self.block_size = 1
#        else:
#            self.block_size = block_size
#
#        self.num_cells = window_size / self.cell_size
#        
#        cix,ciy = np.array(np.meshgrid(np.arange(self.num_cells[1]), np.arange(self.num_cells[0])))
#
#        self.cell_inds = np.array((ciy.ravel(),cix.ravel())).T
#
#        self.pca_model = feature_pca
#        self.pca_dim = pca_dim
#
#        if clustering_method == 'kmeans':
#            self.ff_optype = vivid.FF_OPTYPE_EUCLIDEAN
#        elif clustering_method == 'sp_kmeans':
#            self.ff_optype = vivid.FF_OPTYPE_COSINE
#    
#        self.weighted_hist = int(weighted_hist)
#
#        self.clusters = clusters
#        self.bank_sizes = [len(fb) for fb in self.clusters]
#
#        self.single_cluster = True
#
#        self.n_output_blocks = self.num_cells.prod()
#        self.cell_blocks = np.arange(self.n_output_blocks).reshape((self.n_output_blocks,1))
#
#        if (self.block_normalization == 2) or (self.block_normalization==1): #l2
#            self.cell_blocks = _cell_blocks(self.cell_size, self.window_size, self.block_size)
#            self.n_output_blocks = len(self.cell_blocks) 
#        elif self.block_normalization == 3:  #pytramid
#            self.level_blocks = []
#            self.n_levels = np.floor( np.log(np.min(np.floor(  np.array(self.window_size) / self.cell_size ))) / np.log(2.0)).astype('int')
#
#            self.n_total_blocks = 0
#
#            for li in range(self.n_levels+1):
#                self.level_blocks.append( _cell_blocks(self.cell_size, self.window_size, 2**li, 2**li) )
#                self.n_total_blocks += self.level_blocks[-1].shape[0]
#
#
#    """ For processing multiple scales we just change the origin instead of creating a
#    new feature filter
#    """
#    def update_origin(self,origin):
#        self.origin = origin
#
#    """ This is global L_2 normalization on the entire feature vector
#    """
#    def normalize_features(self, feature_mat):
#        feature_mags = np.sqrt(np.sum(feature_mat*feature_mat,axis=1)) + EPSILON
#        feature_mat = feature_mat / feature_mags[:,np.newaxis]
#
#        return feature_mat   
#
#    """ Projecting to the first pca_dim principal components of each individual cell
#    """
#    def pca_chunk_project(selfm, feature_mat, pca_model, pca_dim):
#        pca_bases = pca_model[1]
#        pca_mean = pca_model[0]
#    
#        dict_size = pca_bases.shape[2]
#    
#        n_chunks = feature_mat.shape[1] / dict_size
#    
#        new_feature_mat = np.zeros((feature_mat.shape[0], n_chunks * pca_dim),dtype='f')
#        
#        feature_mat = feature_mat - pca_mean[np.newaxis,:]
#
#        for chunk_i in range(n_chunks):
#            lo = chunk_i * dict_size
#            hi = (chunk_i+1) * dict_size
#        
#            chunk_fea = np.dot(feature_mat[:,lo:hi], pca_bases[chunk_i,:,:pca_dim])
#            chunk_mag = np.sqrt((chunk_fea * chunk_fea).sum(axis=1))
#            chunk_fea = chunk_fea / chunk_mag[:,np.newaxis]
#    
#            new_feature_mat[:,chunk_i*pca_dim:(chunk_i+1)*pca_dim] = chunk_fea
#
#        return new_feature_mat
#
#    """ Top level function for computing the features on the window list
#    """
#    def get_features(self, frame_num, window_list):
#        n_windows = len(window_list)
#
#        if self.block_normalization > 0:
#            feature_mat = np.zeros((n_windows, self.n_output_blocks, self.dict_size * self.block_size * self.block_size),dtype='float32')
#        else:
#            feature_mat = np.zeros((n_windows, self.n_output_blocks, self.dict_size),dtype='float32')
#
#        max_y = window_list[:,0].max()
#        max_x = window_list[:,1].max()
#
#        mod_y = np.mod(window_list[:,0], self.cell_size)
#        mod_x = np.mod(window_list[:,1], self.cell_size)
#
#        #Determine the x,y offset pairs to be run
#        yx_offsets = []
#        for cy in range (self.cell_size):
#            y_match = mod_y == cy
#            if not np.any(y_match):
#                continue
#            for cx in range(self.cell_size):
#                x_match = mod_x == cx
#                if not np.any(x_match):
#                    continue
#                yx_offsets.append([cy,cx])
#
#        ff = vivid.FlexibleFilter(self.origin, self.clusters[0], self.ff_optype)
#        
#        if self.weighted_hist == 3:
#            frame = self.origin.get_frame(frame_num)
#            frame_size = frame.shape
#            
#            #similarities = convolve_with_filter_bank(frame, self.clusters[0])
#            similarities = ff.filter_frame_cuda_noargmin(frame_num).mat().astype('f')
#            similarities = similarities * similarities
#
#            #do softmax weighting
#            alpha = 10000.0
#            
#            #this is a much faster approximation, it is accurate within the range [-2,2]
#            #exp_w = np.exp(alpha * similarities)
#            exp_w = vivid.fast_exp(alpha * similarities, 5) 
#
#            #similarities = similarities *  exp_w / (exp_w.sum(axis=2))[:,:,np.newaxis]
#            inv_exp_w = (1.0 / (exp_w.sum(axis=2)) )
#            similarities = similarities *  exp_w * inv_exp_w[:,:,np.newaxis]
#
#            del exp_w
#            del inv_exp_w
#        else:
#            assignments, similarities = ff.filter_frame_cuda(frame_num).mat().astype('float32')
#
#            #frame = self.origin.get_frame(frame_num)
#            #similarities = _compute_mean_sub_weight_mat(frame, self.patch_size)
#
#        for yx_offset in yx_offsets:
#            offset_y, offset_x = yx_offset
#            
#            if self.weighted_hist == 3:
#                yl = np.arange(offset_y, frame_size[0]-self.cell_size, self.cell_size)
#                xl = np.arange(offset_x, frame_size[1]-self.cell_size, self.cell_size)
#
#                xi,yi = np.meshgrid(xl,yl)
#
#                cell_hist = np.zeros((yl.size,xl.size,self.dict_size),dtype='f')
#
#                for i in range(self.cell_size):
#                    for j in range(self.cell_size):
#                        cell_hist += similarities[yi+i,xi+j]
#            else:
#                cell_hist = vivid.block_histogram_dense(assignments, similarities, 
#                                                        self.dict_size, self.cell_size,
#                                                        yx_offset)
#            logical_inds = np.logical_and(mod_y==offset_y, mod_x==offset_x)
#
#            start_cell_y = (window_list[logical_inds,0] - offset_y) / self.cell_size
#            start_cell_x = (window_list[logical_inds,1] - offset_x) / self.cell_size
#
#            for bi,cb in enumerate(self.cell_blocks):
#                block_inds = self.cell_inds[cb]
#                biy = block_inds[:,0] + start_cell_y[:,np.newaxis]
#                bix = block_inds[:,1] + start_cell_x[:,np.newaxis]
#
#                loop_hist = cell_hist[biy,bix,:].reshape((start_cell_y.shape[0], self.dict_size * self.block_size * self.block_size))
#
#                if self.block_normalization == 2:
#                    loop_hist_mag = np.sqrt( (loop_hist*loop_hist).sum(axis=1) ) + 1e-7
#                    loop_hist = loop_hist / loop_hist_mag[:,np.newaxis] 
#                elif self.block_normalization == 1:
#                    loop_hist_mag = np.abs(loop_hist).sum(axis=1) + 1e-7
#                    loop_hist = loop_hist / loop_hist_mag[:,np.newaxis] 
#
#                feature_mat[logical_inds,bi,:] = loop_hist
#
#        feature_mat = feature_mat.reshape((feature_mat.shape[0], -1))
#
#        if (self.feature_normalization == 'norm') or (self.feature_normalization == 'sqrt_norm'):
#            feature_mat = self.normalize_features(feature_mat)
#              
#        return feature_mat.reshape((n_windows,-1))


                



#class FeatureSource:
#    def __init__(self,origin, 
#                im_scales=[], 
#                clustering_method='sp_kmeans',
#                patch_type='gradient',
#                clusters = [],
#                cell_size=8,
#                patch_size=3,
#                weighted_hist=2,
#                dict_size=200,
#                window_size=[128,64],
#                block_normalization=2,
#                block_size=2,
#                feature_normalization='norm',
#                window_stride=8,
#                verbose=True,
#                multi_filter_size=3,
#                feature_pca = [],
#                pca_dim = 0):
#
#        self.origin = origin
#        self.verbose = verbose       
#
#        #SET UP BASIC DATA SOURCES
#        gv = vivid.GreyVideo(origin)
#        
#        if im_scales == []:
#            self.im_scales = np.array( [ (1.0/1.2)**s  for s in range(9)] ,dtype='float32')
#        else:
#            self.im_scales = im_scales
#        self.num_scales = len(self.im_scales)
#
#        ps = [vivid.CachedVideo(vivid.ScaledVideo(gv, scale=float(im_sc)),cache_size=1) for im_sc in self.im_scales]
#
#        if patch_type == 'pixel':
#            self.ss = ps
#        elif patch_type == 'mean_sub':
#            self.ss = [vivid.MatSource(p) for p in ps]
#        elif patch_type == 'gradient':
#            self.ss = [vivid.SimpleGradient(p, stacked=True) for p in ps]
#        elif patch_type == 'multi':
#            self.ss = [vivid.MultiFilterSource(p, multi_filter_size=multi_filter_size) for p in ps]
#        elif patch_type == 'multicolor':
#            self.ss = [ vivid.MultiFilterSource(
#                            vivid.CachedVideo(
#                                vivid.ScaledVideo(self.origin, scale=float(im_sc)), cache_size=1), 
#                                multi_filter_size=multi_filter_size) 
#                        for im_sc in self.im_scales]
#
#        #SET UP WINDOW LOCATIONS
#        self.cell_size = cell_size
#        self.window_size = np.array(window_size)
#        self.window_stride = window_stride
#        self.filter_apron = patch_size / 2
#
#        #SET UP THE FLEXIBLE FEATURE SOURCE
#        self.ffs = FlexibleFeatureSource(self.ss[0], clusters,
#                                         dict_size, clustering_method,
#                                         weighted_hist, patch_size, cell_size,
#                                         window_size, block_size,
#                                         block_normalization, feature_normalization,
#                                         feature_pca, pca_dim)
# 
#    def get_cell_locations(self):
#        return _cell_locations(self.cell_size,self.window_size)
#
#    """ No processing here.  It just computes the location of windows in each scale
#    """
#    def get_window_guide(self, frame_num, window_offset=0):
#        window_guide = np.empty((0,3),dtype='float32')
#
#        for si in range(self.num_scales):
#            im_size = self.ss[si].get_frame(frame_num).shape[:2]
#            window_grid = _get_window_grid(im_size, self.window_size, self.filter_apron,  window_offset, stride=self.window_stride)
#            
#            if window_grid == []:
#                continue
#
#            window_list = (np.reshape(np.meshgrid(window_grid[1], window_grid[0]), (2,-1)).T)[:,::-1]
#            scale_guide = np.ones((len(window_list),1)) * self.im_scales[si]
#
#            scale_window_guide = np.hstack((window_list, scale_guide))
#
#            window_guide = np.vstack((window_guide,scale_window_guide)).astype('float32')    
#
#        return window_guide
#
#    """ Returns only the window counts per scale insead of a whole window guide
#    """
#    def get_scale_window_counts(self, frame_num, window_offset=0):
#        scale_window_counts = np.zeros(self.num_scales,dtype='int')
#
#        for si in range(self.num_scales):
#            im_size = self.ss[si].get_frame(frame_num).shape[:2]
#
#            window_grid = _get_window_grid(im_size, self.window_size, self.filter_apron,  window_offset, stride=self.window_stride)
#            
#            if window_grid == []:
#                continue
#            scale_window_counts[si] = len(window_grid[0]) * len(window_grid[1])
#
#        return scale_window_counts
#
#    """ Get all features from a single scale
#    """
#    def get_features_from_scale(self, frame_num, scale=0, window_offset=0, window_inds=[]):
#        self.ffs.update_origin(self.ss[scale])
#        im_size = self.ss[scale].get_frame(frame_num).shape[:2]
#
#        window_grid = _get_window_grid(im_size, self.window_size, 
#                                       self.filter_apron, window_offset, 
#                                       stride=self.window_stride)
#        
#        if window_grid == []:
#            return ([],[])
#        
#        window_list = (np.reshape(np.meshgrid(window_grid[1], window_grid[0]), (2,-1)).T)[:,::-1]
#        if (len(window_inds) != 0):
#            window_list = window_list[window_inds]
#
#        scale_guide = np.ones((len(window_list),1)) * self.im_scales[scale]
#
#        if self.verbose:
#            print "\tscale: %d\tnum_wins: %d"%(scale,len(scale_guide))
#
#        feature_mat = self.ffs.get_features(frame_num, window_list)
#
#        return (feature_mat, np.hstack((window_list, scale_guide)))
#
#    """ Get features from given window indexes
#    Window indexes span over all scales
#    Starting the count from the lowest (i.e., no scaling)
#    """
#    def get_features(self,frame_num,window_offset=0, window_indexes='all'):
#        window_guide = np.empty((0,3),dtype='float32')
#
#        scale_window_counts = self.get_scale_window_counts(frame_num,window_offset)
#        scale_window_cumulative_counts = np.zeros((self.num_scales + 1),dtype='int')
#        scale_window_cumulative_counts[1:] = np.cumsum(scale_window_counts)
#
#        if window_indexes=='all':
#            num_samples = scale_window_counts.sum()
#        else:
#            num_samples = len(window_indexes)
#
#        collected = 0
#        initialized = False
#        all_features = []
#
#        for si in range(self.num_scales):
#            if window_indexes != 'all':
#                inds_in_scale = np.logical_and((window_indexes >= scale_window_cumulative_counts[si]), 
#                                               (window_indexes <  scale_window_cumulative_counts[si+1]))
#                if len(inds_in_scale) == 0:
#                    continue
#                wins_in_scale = window_indexes[inds_in_scale]
#                wins_in_scale -= scale_window_cumulative_counts[si]
#
#            feature_mat, scale_window_guide = self.get_features_from_scale(frame_num, si, window_offset)
#            if feature_mat == []:
#               break
#
#            if not initialized:
#                feature_dim = feature_mat.shape[1]
#                all_features = np.zeros((num_samples, feature_dim),dtype='float32')
#                initialized = True
#
#            if window_indexes != 'all':
#                scale_window_guide = scale_window_guide[wins_in_scale]
#                feature_mat = feature_mat[wins_in_scale]
#
#            window_guide = np.vstack((window_guide,scale_window_guide)).astype('float32')    
#            all_features[collected:collected+feature_mat.shape[0],:] = feature_mat
#
#            collected += feature_mat.shape[0]
#
#        return (all_features, window_guide)
