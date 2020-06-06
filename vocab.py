#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import re
import string
import spacy
nlp_w = spacy.load('en',disable=['parser', 'tagger', 'ner'])
nlp_s = spacy.load('en',disable=['tagger', 'ner'])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import torch
import torchtext
import torchtext.data as data
from functools import partial
import revtok
import os
import glob
from progiter import ProgIter as tqdm
import random
import time
import dill
from collections import Counter, OrderedDict
import argparse
from pytorch_pretrained_bert import BertTokenizer
from sample_onet_classes import onet_taxononomy


'''
Create torchtext vocab object
'''
    
class vocab():
    
    def __init__(self):
        self.normal = partial(torch.nn.init.normal_, mean=0, std=0.05)
        self.onet_hash = pickle.load(open('./Data/data_new/onet_hash', 'rb'))
            
        self.job_desc_vocab = Counter()
        self.hash_id_vocab = Counter()
        # self.ob = Counter()
        self.onet_desc_vocab = Counter()
        self.job_desc_processed_dict = dict()
        self.job_title_dict = pickle.load(open('./Data/data_new/job_title_dict', 'rb'))
    
    def data_gen_softmax(self, category):
        global in_dir
        max_batches=1000
        for batch_ind in tqdm(range(max_batches)):
            # print(category, batch_ind)
            try:
                data = pickle.load(open(f'./Data/data_new/{in_dir}/{category}/data_{batch_ind}.pkl', 'rb'))
            except:
                break
            for i, (hash_id, job_desc) in enumerate(data):
                job_title = self.job_title_dict[hash_id]
                onet_code = self.onet_hash[hash_id]
                yield (hash_id, job_title, job_desc, onet_code)
                
                
    def build_vocab(self):
        TEXT = data.ReversibleField()
        nestedfield = data.NestedField(nesting_field=TEXT,
                                      include_lengths=True,)
        HASH = data.ReversibleField(sequential=False, use_vocab=True)
        ONET_CODE = data.ReversibleField(sequential=False, use_vocab=True, is_target=True)
        
        word_vectors = 'fasttext.en.300d'
        
        
        for category in ['train', 'test', 'unseen']:
            for hash_id, job_title, job_desc, onet_code in self.data_gen_softmax(category):
#                 self.job_desc_processed_dict[hash_id] = job_desc
                self.hash_id_vocab.update([hash_id,])
                self.job_desc_vocab.update(job_title)
                for sentence in job_desc:
                    self.job_desc_vocab.update(sentence)                
        
        self.job_desc_vocab = torchtext.vocab.Vocab(self.job_desc_vocab, vectors=word_vectors)
        with open('./Data/data_new/job_desc_vocab', 'wb') as handle: 
            dill.dump(self.job_desc_vocab, handle)
#         TEXT.vocab = dill.load(open('./Data/data_new/job_desc_vocab', 'rb'))
       
        self.onet_code_vocab = torchtext.vocab.Vocab(self.onet_code_vocab)
        with open('./Data/data_new/onet_code_vocab', 'wb') as handle: 
            dill.dump(self.onet_code_vocab, handle)
#         ONET_CODE.vocab = dill.load(open('./Data/data_new/onet_code_vocab', 'rb'))

        self.hash_id_vocab = torchtext.vocab.Vocab(self.hash_id_vocab)
        with open('./Data/data_new/hash_id_vocab', 'wb') as handle: 
            dill.dump(self.hash_id_vocab, handle)
#         HASH.vocab = dill.load(open('./Data/data_new/hash_vocab', 'rb'))
        

if __name__=="__main__":
    global in_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, help='Input directory containing batches', default='sentenced_job_desc_sample')
    args = parser.parse_args()
    in_dir = args.in_dir
    vocab = vocab(bert=False)
    vocab.build_vocab()


# In[ ]:




