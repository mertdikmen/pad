from sys import version_info
if version_info.major == 2:
    import ConfigParser as configparser
    import cPickle as pickle
else:
    import configparser
    import pickle
import numpy as np
import os

class DictionaryOpt(object):
    def __init__(self, clustering_method, size):
        self.clustering_method = clustering_method
        self.size = size

class CellOpt(object):
    def __init__(self, patch_size=3, cell_size=8, distance_type='abs_cosine',
                 dictionary_size=200, weighted_hist='l2'):
        self.patch_size = patch_size
        self.cell_size = cell_size
        self.dictionary_size = dictionary_size
        self.distance_type = distance_type
        self.weighted_hist =  weighted_hist

class BlockOpt(object):
    def __init__(self,
                 block_size=2,
                 block_normalization='l2',
                 cell_grouping='concatenate'):
        self.block_size = block_size
        self.block_normalization = block_normalization
        self.cell_grouping = cell_grouping

class FeatureOpt(object):
    def __init__(self,
                 window_size, fea_type,
                 scale_step=1.2,
                 feature_normalization='l2',
                 window_stride=4,
                 patch_size=3,
                 cell_size=8,
                 distance_type='abs_cosine',
                 dictionary_size=200,
                 weighted_hist='l2',
                 block_size=2,
                 block_normalization='l2',
                 cell_grouping='concatenate'):

        self.fea_type = fea_type

        self.cell = CellOpt(patch_size,
                            cell_size,
                            distance_type,
                            dictionary_size,
                            weighted_hist)

        self.block = BlockOpt(block_size,
                              block_normalization,
                              cell_grouping)

        self.scale_step = scale_step

        self.feature_normalization = feature_normalization
        self.window_stride = window_stride
        self.window_size = np.array(window_size)

        self.blocks_per_win = (self.window_size / self.cell.cell_size -
                               self.block.block_size + 1)

        if self.block.cell_grouping == 'concatenate':
            self.feature_dim = (self.blocks_per_win[0] * self.blocks_per_win[1]
                              * self.block.block_size * self.block.block_size
                              * self.cell.dictionary_size)
        elif self.block.cell_grouping == 'add':
            self.feature_dim = (self.cell.dictionary_size *
                self.blocks_per_win[0] * self.blocks_per_win[1])


class ConfigOpt(object):
    def get_print_options(self):
        opt_text = ""
        opt_text+= "Config file:\t %s\n\n"%self.source_file

        opt_text+= "Dataset root:\t %s\n"%self.dataset_root
        opt_text+= "Output root:\t %s\n"%self.output_root
        opt_text+= "Image root:\t %s\n\n"%self.image_root

        opt_text+= "Stash path:\t %s\n"%self.stash_path
        opt_text+= "Feature path:\t %s\n"%self.feature_path
        opt_text+= "SVM Model path:\t %s\n"%self.svm_model_path
        opt_text+= "Result path:\t %s\n\n"%self.result_path

        opt_text+= "Patch file:\t %s\n"%self.patch_file
        opt_text+= "Num patches:\t %d\n"%self.num_patches
        opt_text+= "Seed set:\t %s\n\n"%self.seed_set

        opt_text+= "Object size:\t %dx%d\n\n"%(self.object_size[0],
                                               self.object_size[1])

        opt_text+= "Dictionary Options\n"
        opt_text+= "\tDictionary file:\t %s\n"%self.dictionary_file
        opt_text+= "\tClustering method:\t %s\n"%(
            self.dictionary.clustering_method)
        opt_text+= "\tDictionary size:\t %d\n"%self.dictionary.size

        opt_text+= "\nFeature Options\n"
        opt_text+= "\tFeature type: {0}\n".format(self.feature.fea_type)
        opt_text+= "\tBlocks per window:\t %dx%d\n"%(
            self.feature.blocks_per_win[0], self.feature.blocks_per_win[1])
        opt_text+= "\tFeature dimension:\t %d\n"%self.feature.feature_dim
        opt_text+= "\tFeature normalization:\t %s\n"%(
            self.feature.feature_normalization)

        opt_text+= "\tTraining samples per image:\t %d\n"%(
            self.training_samples_per_im)

        opt_text+= "\n\tBlock Options\n"
        opt_text+= "\t\tNorm:\t %s\n"%self.feature.block.block_normalization
        opt_text+= "\t\tSize:\t %d\n"%self.feature.block.block_size
        opt_text+= "\t\tGrouping:\t %s\n"%self.feature.block.cell_grouping

        opt_text+= "\n\tCell Options\n"
        opt_text+= "\t\tCell Size:\t %d\n"%self.feature.cell.cell_size
        opt_text+= "\t\tDistance:\t %s\n"%self.feature.cell.distance_type
        opt_text+= "\t\tWeighted hist:\t %s\n"%self.feature.cell.weighted_hist
        opt_text+= "\t\tDictionary Size: %d\n"%self.feature.cell.dictionary_size
        opt_text+= "\t\tPatch Size:\t %d\n"%self.feature.cell.patch_size  

        return opt_text

    def print_options(self):
        print(self.get_print_options())

    def __init__(self,source_file):
        self.source_file = source_file
        config = configparser.ConfigParser()

        config.read(self.source_file)

        #: Root of the dataset
        self.dataset_root = config.get("Dataset", "root") 

        #: Root of the images
        self.image_root = config.get("Dataset", "image_root") 

        default_image_suffix = config.get("Dataset", "default_image_suffix") 
        training_list_file = os.path.join(
            self.dataset_root, config.get("Dataset", "TrainingFileList"))
        validation_list_file = os.path.join(
            self.dataset_root, config.get("Dataset", "ValidationFileList")) 
        #evaluation_list_file = os.path.join( self.dataset_root, config.get("Dataset", "EvaluationFileList") )

        self.image_sets = {'training': [], 'validation': []}
        self.image_sets_fp = {'training': [], 'validation': []}

        with open(training_list_file, 'rb') as tf:
            #: List of images in the training set
            for t in tf.read().splitlines():
                if version_info.major == 3:
                    t = str(t, encoding='utf8')
                file_name  = t.split(' ')[0]
                suffix_check = len(file_name.split('.')) == 2

                if suffix_check:
                    self.image_sets['training'].append(file_name)
                else:
                    self.image_sets['training'].append(file_name + '.' +
                                                       default_image_suffix)

            #: full paths of images in the training set
            self.image_sets_fp['training'] = [
                os.path.join(self.image_root, t) 
                for t in self.image_sets['training']] 

        with open(validation_list_file, 'rb') as tf:
            #: full paths of images in the validation set
            for t in tf.read().splitlines():
                if version_info.major == 3:
                    t = str(t, encoding='utf8')
                file_name  = t.split(' ')[0]
                suffix_check = len(file_name.split('.')) == 2

                if suffix_check:
                    self.image_sets['validation'].append(file_name)
                else:
                    self.image_sets['validation'].append(file_name + '.' +
                                                         default_image_suffix)

            #: full paths of images in the validation set
            self.image_sets_fp['validation'] = [
                os.path.join(self.image_root, t)
                for t in self.image_sets['validation']]  

        #: Number of patches to collect
        self.num_patches = config.getint("PatchExtraction", "NumPatches") 

        #: Which set to use for random patch collection
        self.seed_set = config.get("PatchExtraction", "SeedSet") 

        #: Size of the object
        self.object_size = [
            config.getint("Object", "size_v"),
            config.getint("Object", "size_h")]

        self.training_samples_per_im = config.getint(
            "Training", "training_samples_per_im")

        #: FeatureOpt object.  Contains the feature options.
        self.feature = FeatureOpt(
                        self.object_size,
                        config.get("Feature", "type"),
                        config.getfloat("Feature", "scale_step"),
                        config.get("Feature", "feature_normalization"),
                        config.getint("Feature", "window_stride"),
                        config.getint("Feature", "patch_size"),
                        config.getint("Feature", "cell_size"),
                        config.get("Feature", "distance_type"),
                        config.getint("Dictionary", "size"),
                        config.get("Feature", "weighted_hist"),
                        config.getint("Feature", "block_size"),
                        config.get("Feature", "block_normalization"),
                        config.get("Feature", "cell_grouping"))

        #: String to be appended to the feature files
        self.feature_str = self.feature.fea_type + "."

        with open(config.get("Annotations", "annotation_file"),'rb') as ann_file:
            #: All annotations
            self.annotations = pickle.load(ann_file)

        #: Indexes of the annotations 
        self.annotation_inds = {'training': [], 'validation': []}

        #: Indexes of image files of the annotations 
        self.annotation_image_inds = {'training': [], 'validation': []}

        n_missed_images = 0
        for i,annotation in enumerate(self.annotations):
            try: #hit the training set first
                image_ind = self.image_sets['training'].index(
                    annotation.image_name)
                self.annotation_inds['training'].append(i)
                self.annotation_image_inds['training'].append(image_ind)
            except ValueError:
                try:
                    image_ind = self.image_sets['validation'].index(
                        annotation.image_name)
                    self.annotation_inds['validation'].append(i)
                    self.annotation_image_inds['validation'].append(image_ind)
                except ValueError:
                   #print("{0} is neither in training nor in validation".format(
                   #    annotation.image_name))
                   n_missed_images += 1

        if n_missed_images > 0:
            print("{0}".format(n_missed_images),
                  "images neither in training nor in validation sets")

        #: Indexes of images containing neither testing or validation annotations
        self.non_annotation_image_inds = {
            'training': list(set(range(len(self.image_sets['training']))).difference(self.annotation_image_inds['training'])),
            'validation': list(set(range(len(self.image_sets['validation']))).difference(self.annotation_image_inds['validation']))
        }

        #: Output directory for end and intermediate results
        self.output_root = config.get("Output","root") 
        self.stash_path = os.path.join(self.output_root, "STASH")

        #: File for storing random pathces
        self.patch_path = os.path.join(self.output_root, "patches")
        try:
            os.makedirs(self.patch_path)
        except:
            pass

        self.patch_file = os.path.join(
            self.patch_path,
            "seed.%s_n.%g.npy"%(self.seed_set, self.num_patches)) 

        #: File for storing the dictionary
        self.dictionary = DictionaryOpt(
            config.get("Dictionary", "clustering_method"),
            config.getint("Dictionary", "size"))

        self.dictionary_path = os.path.join(self.output_root, "dictionaries")

        try:
            os.makedirs(self.dictionary_path)
        except:
            pass

        self.dictionary_file = os.path.join(self.dictionary_path,
                                            "seed.%s_%s_d.%04d.npy"%(self.seed_set, 
                                                                     self.dictionary.clustering_method, 
                                                                     self.dictionary.size)) 

        #: Feature target path
        self.feature_path = os.path.join(self.output_root, "features")
        try:
            os.makedirs(self.feature_path)
        except:
            pass

        #: SVM Model Path
        self.svm_model_path = os.path.join(self.output_root, "svm_models")
        try:
            os.makedirs(self.svm_model_path)
        except:
            pass

        #: Result path
        self.result_path = os.path.join(self.output_root, "results")
        try:
            os.makedirs(self.result_path)
        except:
            pass

        self.print_options()
