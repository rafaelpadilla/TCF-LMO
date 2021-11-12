import os

# Change here the path of the dataset
# Notice: Geometric alignment was not used in the paper
base_dir_geo_align_lmdb = './dataset/'
DIR_GEOMETRIC_TRAIN_LMDB = os.path.join(base_dir_geo_align_lmdb, 'geo_align_train_lmdb')
DIR_GEOMETRIC_VAL_TEST_LMDB = os.path.join(base_dir_geo_align_lmdb, 'geo_align_val_test_lmdb')

base_dir_temporal_align_lmdb = './dataset/'
DIR_TEMPORAL_TRAIN_LMDB = os.path.join(base_dir_temporal_align_lmdb, 'training_ds',
                                       'temporal_align_train_lmdb')
DIR_TEMPORAL_VAL_TEST_LMDB = os.path.join(base_dir_temporal_align_lmdb, 'testing_ds',
                                          'temporal_align_val_test_lmdb')
