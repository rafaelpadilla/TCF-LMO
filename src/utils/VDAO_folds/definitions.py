import enum
import os
import socket

################################################################################
#                            UNIVERSAL DEFINITIONS                             #
################################################################################
random_seed = 123
current_dir = os.path.dirname(os.path.abspath(__file__))

hostname = socket.gethostname()
current_path = os.path.dirname(os.path.abspath(__file__))
if hostname == 'notesmt':
    DATABASE_DIR = '/media/storage/VDAO'
    ALIGNMENT_DIR = '/media/storage/VDAO'
elif 'ufrj.br' in hostname:
    DATABASE_DIR = '/home/rafael.padilla/workspace/rafael.padilla'
    ALIGNMENT_DIR = '/home/rafael.padilla/nfs/'

videos_dir = {
    'train': os.path.join(DATABASE_DIR, 'vdao_object'),
    'validation': os.path.join(DATABASE_DIR, 'vdao_research'),
    'test': os.path.join(DATABASE_DIR, 'vdao_research')
}

csv_dir = {
    'train':
    os.path.join(ALIGNMENT_DIR, 'vdao_alignment_object/shortest_distance/intermediate_files/'),
    'validation':
    os.path.join(ALIGNMENT_DIR, 'vdao_alignment_research/shortest_distance/intermediate_files/'),
    'test':
    os.path.join(ALIGNMENT_DIR, 'vdao_alignment_research/shortest_distance/intermediate_files/')
}

PATH_JSON_TRAIN_VAL_TEST = os.path.join(current_dir, 'train_val_test.json')
PATH_JSON_FOLDS = os.path.join(current_dir, 'all_folds.json')

target_objects = ['black backpack', 'black coat', 'brown box', 'camera box', 'dark-blue box', 'pink bottle', 'shoe', 'towel', 'white jar']

all_layers = [
    'conv1', 'residual1', 'residual2', 'residual3', 'residual4', 'residual5', 'residual6',
    'residual7', 'residual8', 'residual9', 'residual10', 'residual11', 'residual12', 'residual13',
    'residual14', 'residual15', 'residual16'
]

class JSON_file(enum.Enum):
    Object = os.path.join(current_dir, 'vdao_object.json')
    Research = os.path.join(current_dir, 'vdao_research.json')

class Type_dataset(enum.Enum):
    Train = 1
    Validation = 2
    Test = 3

class Video_type(enum.Enum):
    Reference = 1
    Target = 2
