#!/usr/bin/env python
#-*- coding:utf8 -*-
import os
import json
import random
from utils import ProgressBar, get_time


class DataLoader(object):
    def __init__(self, label_tags, source_path, logid_path):
        self._label_tags = label_tags
        self._source_path = source_path
        self._logid_path = logid_path
        self._feature_data = list()
        self._label_data = list()
        self._data_ids = list()

    @property
    def logid_dict(self):
        logid_dict = dict()
        with open(self._logid_path) as f:
            for i, line in enumerate(f):
                logid_dict[line.strip()] = i+2
        return logid_dict

    def _load(self):
        for i, label_tag in enumerate(self._label_tags):
            path = os.path.join(self._source_path, label_tag)
            print('\nLoading {} data...'.format(label_tag))
            pbar = ProgressBar(len(os.listdir(path)))
            for j, filename in enumerate(os.listdir(path)):
                filepath = os.path.join(path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        # print filepath
                        feature_seq = json.load(f)
                        feature = self._sequence2feature(feature_seq)
                        self._data_ids.append(filepath.split('/')[-1])
                        self._feature_data.append(feature)
                        self._label_data.append(i)
                except EOFError:
                    print('[{time}] Failed to load file {filepath}'
                        .format(time=get_time(), filepath=filepath))
                pbar.updateBar(j)

    def _shuffle(self):
        samp = random.sample(range(len(self._label_data)), len(self._label_data))
        self._feature_data = [self._feature_data[idx] for idx in samp]
        self._label_data = [self._label_data[idx] for idx in samp]

    def _sequence2feature(self, feature_seq):
        # feature = [0] * (len(self.logid_dict) + 1)
        # for item in feature_seq:
        #     feature[int(self.logid_dict[item]) - 1 if item in self.logid_dict else 0] += 1
        feature = [0] * 431
        for item in feature_seq:
             feature[int(item) - 1] += 1
        return feature

    def run(self):
        self._load()
        self._shuffle()