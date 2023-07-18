from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from torchvision.transforms import transforms


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None, target_transform=None, float_precision=32):
        assert float_precision in (32, 64), "Required 32 or 64 but {} is given".format(float_precision)
        assert split in ('train', 'test'), "Required 'train' or 'test but {} is given".format(split)
        if float_precision == 32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64
        self.float_precision = float_precision
        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
        self.split = split
        
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        # self.captions = self.load_all_captions()
        if split == "train":
            self.filenames = self.load_filenames(split_dir)
            self.class_id = self.load_class_id(split_dir, len(self.filenames))
    
    def get_img(self, img_path, bbox) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        # load_size = int(self.imsize * 76 / 64)
        # img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
        return img.type(self.dtype)
    
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
    
    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict
    
    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions
    
    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'
        elif os.path.isfile(os.path.join(data_dir, embedding_type)):
            # embeddings are provided as files
            embedding_filename = embedding_type
        else:
            raise ValueError("No embedding files was found '{}'".format(embedding_type))
        
        # > https://github.com/reedscot/icml2016
        # > https://github.com/reedscot/cvpr2016
        # > https://arxiv.org/pdf/1605.05395.pdf
        
        with open(os.path.join(data_dir, embedding_filename), 'rb') as f:
            embeddings = pickle.load(f, encoding="bytes")
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape, "original dtype:", embeddings.dtype)
        return torch.tensor(embeddings, dtype=self.dtype)
    
    def load_class_id(self, data_dir, total_num):
        path_ = os.path.join(data_dir, 'class_info.pickle')
        if os.path.isfile(path_):
            with open(path_, 'rb') as f:
                class_id = np.array(pickle.load(f, encoding="bytes"))
        else:
            class_id = np.arange(total_num)
        print('Class_ids: ', class_id.shape, "Sample:", class_id[0])
        return class_id
    
    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)), "sample:", filenames[0])
        return filenames
    
    def __getitem__(self, index):
        # captions = self.captions[filepath]
        embeddings = self.embeddings[index, :, :]
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        if self.split == "train":
            filepath = self.filenames[index]
            # cls_id = self.class_id[index]
            if self.bbox is not None:
                bbox = self.bbox[filepath]
                data_dir = '%s/CUB_200_2011' % self.data_dir
            else:
                bbox = None
                data_dir = self.data_dir
            
            img_name = os.path.join(data_dir, filepath)
            assert os.path.isfile(img_name), img_name
            img = self.get_img(img_name, bbox)
            return img, embedding
        else:
            return embedding
    
    def __len__(self):
        return len(self.filenames)
