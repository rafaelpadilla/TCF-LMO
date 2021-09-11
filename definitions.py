import os

# Change here the path of the dataset
base_dir_geo_align_lmdb = '/nfs/proc/rafael.padilla/geo_align_lmdb'
DIR_GEOMETRIC_TRAIN_LMDB = os.path.join(base_dir_geo_align_lmdb, 'geo_align_train_lmdb')
DIR_GEOMETRIC_VAL_TEST_LMDB = os.path.join(base_dir_geo_align_lmdb, 'geo_align_val_test_lmdb')
base_dir_temporal_align_lmdb = '/nfs/proc/rafael.padilla/temporal_align_lmdb'
DIR_TEMPORAL_TRAIN_LMDB = os.path.join(base_dir_temporal_align_lmdb, 'temporal_align_train_lmdb')
DIR_TEMPORAL_VAL_TEST_LMDB = os.path.join(base_dir_temporal_align_lmdb,
                                          'temporal_align_val_test_lmdb')
