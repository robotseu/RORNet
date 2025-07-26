import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data as Data
from torchvision import transforms

import codes.models.encoder.clip.clip as clip

from PIL import Image

def get_tac_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]))
    return transforms.Compose(transform_list)

class TacGraspDataSet(Data.Dataset):
    def __init__(self, cfg, preprocess):
        super(TacGraspDataSet, self).__init__()

        self.split = cfg["dataset"]["split"]
        self.input_shape = cfg["dataset"]["input_shape"]
        self.anns_path = cfg["dataset"]["anns_path"]
        self.image_path = cfg["dataset"]["img_path"]
        self.transforms = preprocess
        self.tac_transforms = get_tac_transform()

        # 读取数据集文本
        if self.split == "train":
            with open(self.anns_path["train"]) as f:
                lines = f.readlines()
        else:
            with open(self.anns_path["test"]) as f:
                lines = f.readlines()
        self.lines = lines
        self.data_size = len(self.lines)
        print('Dataset size:', self.data_size)

    def load_refs_feats(self, idx):
        line_data = self.lines[idx].split()
        stop = len(line_data)
        for i in range(1, len(line_data)):
            if line_data[i] == '~':
                stop = i
                break
        sentences = []
        sent_stop = stop + 1
        for i in range(stop + 1, len(line_data)):
            if line_data[i] == '~':
                sentences.append(line_data[sent_stop:i])
                sent_stop = i + 1
        sentences.append(line_data[sent_stop:len(line_data)])
        choose_index = np.random.choice(len(sentences))
        sent = sentences[choose_index]

        ref = ""
        for i in range(0, len(sent)):
            ref = ref + sent[i] + " "

        ref_feat = clip.tokenize([ref])

        return ref_feat

    def load(self, idx):
        line_data = self.lines[idx].split()

        ref = self.load_refs_feats(idx)

        label = line_data[1]

        seq_path = os.path.join(self.image_path, line_data[0])

        img_path = os.path.join(seq_path, 'img_0.png')
        img = Image.open(img_path)
        img = self.transforms(img)

        tac_path = os.path.join(seq_path, 'tac_0.png')
        tac_img = cv2.imread(tac_path, cv2.IMREAD_UNCHANGED)
        tac_img = cv2.resize(tac_img, (224, 224))
        tac_img = self.tac_transforms(tac_img)

        img_seq = img.unsqueeze(1)
        tac_seq = tac_img.unsqueeze(1)

        for i in range(1, 8):
            img_path = os.path.join(seq_path, 'img_{}.png'.format(i))
            img = Image.open(img_path)
            img = self.transforms(img)
            img = img.unsqueeze(1)
            img_seq = torch.cat([img_seq, img], dim=1)

            tac_path = os.path.join(seq_path, 'tac_{}.png'.format(i))
            tac_img = cv2.imread(tac_path, cv2.IMREAD_UNCHANGED)
            tac_img = cv2.resize(tac_img, (224, 224))
            tac_img = self.tac_transforms(tac_img)
            tac_img = tac_img.unsqueeze(1)
            tac_seq = torch.cat([tac_seq, tac_img], dim=1)

        return ref, img_seq.transpose(0, 1), tac_seq.transpose(0, 1), label

    def __getitem__(self, idx):
        ref, img_seq, tac_seq, label = self.load(idx)

        return ref.long(), img_seq, tac_seq, label

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)