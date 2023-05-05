from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6

__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = ''
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 64
__C.STAGE = 1


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024

import sys


def is_python_version(major, minor=None) -> bool:
    """
    Check for specific python major version and optionally minor version

    Args:
        major: int
        minor: int [optional]

    Return:
        True is major[and minor] version matched with installed Python
    """
    assert isinstance(major, int)
    if minor is None:
        return sys.version_info[0] == major
    else:
        assert isinstance(minor, int)
        return sys.version_info[0] == major and sys.version_info[1] == minor


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
    if is_python_version(2):
        dict_iter = a.iteritems
    elif is_python_version(3):
        dict_iter = a.items
    else:
        return

    for k, v in dict_iter():
        # a must specify keys that are in b
        if (is_python_version(2) and not b.has_key(k)) or (is_python_version(3) and k not in b):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        if is_python_version(2):
            yaml_cfg = edict(yaml.load(f))
        elif is_python_version(3):
            yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, __C)
