import numpy as np
import sys
import vivid

from config import ConfigOpt
from opt import options

split_size = 1e4

max_iter = 1000
break_thresh = 1e-1

config_opt = ConfigOpt(options.config_source)

source_file = config_opt.patch_file

target_file = config_opt.dictionary_file

print "source file: %s"%source_file
print "target file: %s"%target_file

try:
    os.makedirs(os.path.join(config_opt.output_root, "dictionaries"))
except:
    pass

all_patches = np.load(source_file)

#center the data
all_patches -= all_patches.mean(axis=1)[:,np.newaxis]

#filter small magnitude
patch_mags = np.sqrt(np.sum(all_patches * all_patches, axis=1))
valid_mags = patch_mags >= 1e-1
patch_mags = patch_mags[valid_mags]

#set magnitude = 1
all_patches = all_patches[valid_mags] / patch_mags[:,np.newaxis]

if config_opt.dictionary.clustering_method == 'kmeans':
    k = vivid.Kmeans(all_patches, config_opt.feature.cell.dictionary_size, split_size)
    fitness = np.Inf

elif config_opt.dictionary.clustering_method == 'sp_kmeans':
    k = vivid.Kmeans(all_patches, config_opt.feature.cell.dictionary_size, 10000, dist_fun='abscosine', centering='medoid')
    fitness = 0

print 'starting iterations'
for iteration in range(max_iter):
    k.iterate()
    cur_fitness = k.fitness
    fit_diff = fitness - cur_fitness
    print "iter: %d\tvalue: %.2f\tdiff:%.3f"%(iteration, cur_fitness, fit_diff)
    if (iteration > 0)  and (np.abs(fit_diff) < break_thresh):
        break
    fitness = cur_fitness
    sys.stdout.flush()

if iteration == max_iter - 1:
    print "Terminated because the maximum number of iterations is reached."

res_dict_size = len(k.centers)

print "Resulting dictionary size %d"%res_dict_size

valid_ind = np.empty(res_dict_size,dtype='bool')
valid_ind.fill(True)

for i,di in enumerate(k.centers):
    if np.allclose(di,0):
        valid_ind[i] = False

dictionary = k.centers[valid_ind]

np.save(target_file, dictionary)

##    #spherical k-means needs normalization
##    all_patches = all_patches.astype('float32')
##    patch_mags = np.sqrt(np.sum(all_patches * all_patches,axis=1))
##    all_patches = all_patches[patch_mags > 0]
##    all_patches[all_patches[:,0] < 0] = -all_patches[all_patches[:,0] < 0]
## 
##    feature_dim = all_patches.shape[1]
##    n_samples = all_patches.shape[0]    
##
##    n_binary_combos = 2**(feature_dim-1) #last dim is always 0
##
##    combo_sizes = np.zeros(n_binary_combos)
##
##    rand_init_centers = []
##
##    for i in range(0,n_binary_combos):
##        binary_str = np.binary_repr(i)
##        init = False
##        for j in range(feature_dim-1):
##            binary_ind = 2**j
##            if np.bitwise_and(i, binary_ind)== binary_ind: #j'th bit is 1
##                cur_ind = (all_patches[:,j] >= 0).astype('int')
##            else:
##                cur_ind = (all_patches[:,j] < 0).astype('int')
##
##            if not init:
##                ind = cur_ind
##                init = True
##            else:
##                ind = cur_ind * ind
##
##        indexes = np.nonzero(ind)
##
##        if len(indexes[0]) != 0:
##            random_pick = np.random.randint(len(indexes[0]))
##            random_pick = indexes[0][random_pick]
##            rand_init_centers.append(all_patches[random_pick])
##                   
##    rand_init_centers = np.array(rand_init_centers)
##    rand_init_mags = np.sqrt((rand_init_centers * rand_init_centers).sum(axis=1))
##    rand_init_centers = rand_init_centers / rand_init_mags[:,np.newaxis]
##
##    k.centers[:rand_init_centers.shape[0]] = rand_init_centers
##    k.centers_DM = vivid.DeviceMatrix(k.centers)
