#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import re
import string


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
from collections import Counter, defaultdict
import pickle

import itertools

from sample_onet_classes import onet_taxononomy
from vocab import get_onet_descriptions, get_job_title_dict
from vocab import onet_desc_tokenizer
from process_raw_batches import standard_preprocess
from sklearn.neighbors import NearestNeighbors
import random
random.seed(32)
# %%pixie_debugger

class dataiter():
    
    def __init__(self, T_depth=4, directory='sentenced_job_desc_sample'):
        
        self.directory = directory
        random.seed(42)
        self.random_state = random.getstate()
        self.batch_size = 2048
        self.onet_hash = pickle.load(open('./Data/data_new/onet_hash', 'rb'))
        self.onet_subset, self.onet_seen, self.onet_unseen = pickle.load(open(f'./Data/data_new/{self.directory}/onet_sampled', 'rb')) 
        self.onet_codes_freq_dict = dict()
        #for onet_code in (self.onet_seen+self.onet_unseen):
        self.onet_codes_freq_dict = defaultdict(int)
        self.onet_codes_dict, self.onet_codes_children, self.onet_codes_neighbors = onet_taxononomy(self.onet_seen)
        self.onet_code_description_dict = get_onet_descriptions()
        self.onet_information_dict = pickle.load(open('./Data/data_new/onet_information_dict', 'rb'))
        self.process_onet_info()
        self.T_depth = T_depth
        if(not os.path.exists('./Data/data_new/job_title_dict')):
            get_job_title_dict()
        self.job_title_dict = pickle.load(open('./Data/data_new/job_title_dict', 'rb'))
        

    def batch_size_fn(self, x, count, sofar):
        """
        Limit total number of documents in batch to 64
        Limit total number of sentences in batch to self.batch_size (=2048)
        """
        if(count > 64):
            return self.batch_size
        return sofar + len(x.job_desc)
    

    def data_gen_softmax(self, batch_ind, DOC_IND, HASH, nestedfield, ONET_CODE, ONET_CODE_l3, ONET_CODE_l2, ONET_CODE_l1, category='train'):
        """
        Convert raw data containing job title, job description, onet code and onet description into format required by torchtext
        """

        examples = []
        try:
            data_batch = pickle.load(open(f'./Data/data_new/{self.directory}/{category}/data_{batch_ind}.pkl', 'rb'))
        except:
            return []
        for i, (hash_id, job_desc) in enumerate(data_batch):
            
            job_desc = [self.job_title_dict[hash_id],] + job_desc
            job_desc = [sentence for sentence in job_desc if sentence!=[]]
            if(len(job_desc) < 3):
                continue
            # assert hash_id == hash_id_bert
            onet_code = self.onet_hash[hash_id]
            
            onet_desc = self.onet_information_dict[onet_code]['Tasks']
            onet_desc = [elem[1] for elem in onet_desc]
            # onet_title = self.onet_information_dict[onet_code]['OnetTitle']
            # onet_desc_p = [onet_title,]+onet_desc_p

            onet_code_l3 = int(onet_code/1000)*1000
            onet_code_l2 = int(onet_code/100000)*100000
            onet_code_l1 = int(onet_code/1000000)*1000000
            if(str(onet_code_l3) not in ONET_CODE_l3.vocab.freqs.keys()):
                continue
            self.onet_codes_freq_dict[onet_code]+=1
            example = data.Example()
            example = example.fromlist([i, hash_id, job_desc, str(onet_code), str(onet_code_l3), str(onet_code_l2), 
                str(onet_code_l1), onet_desc], [('doc_ind', DOC_IND), ('hash', HASH), ('job_desc', nestedfield), 
                ('onet_code', ONET_CODE), ('onet_code_l3', ONET_CODE_l3), ('onet_code_l2', ONET_CODE_l2), ('onet_code_l1', ONET_CODE_l1), 
                ('onet_desc', nestedfield)])
            examples.append(example) 
        return examples
              

    def get_iters_softmax(self):
        """
        Split data into train test and return corresponding iterators.
        """

        TEXT = data.ReversibleField()
        nestedfield = data.NestedField(nesting_field=TEXT,
                                      include_lengths=True,)

        DOC_IND = data.Field(sequential=False, dtype=torch.long, use_vocab=False)
        HASH = data.ReversibleField(sequential=False, use_vocab=True)
        ONET_CODE = data.ReversibleField(sequential=False, use_vocab=True, is_target=True)
        ONET_CODE_l3 = data.ReversibleField(sequential=False, use_vocab=True, is_target=True)
        ONET_CODE_l2 = data.ReversibleField(sequential=False, use_vocab=True, is_target=True)
        ONET_CODE_l1 = data.ReversibleField(sequential=False, use_vocab=True, is_target=True)
        
        TEXT.vocab = dill.load(open('./Data/data_new/job_desc_vocab', 'rb'))
        ONET_CODE.vocab = torchtext.vocab.Vocab(Counter([str(onet_code) for onet_code in self.onet_seen]), specials=[])
        ONET_CODE_l3.vocab = torchtext.vocab.Vocab(Counter([str(int(onet_code/1000)*1000) for onet_code in self.onet_seen]), specials=[])
        ONET_CODE_l2.vocab = torchtext.vocab.Vocab(Counter([str(int(onet_code/100000)*100000) for onet_code in self.onet_seen]), specials=[])
        ONET_CODE_l1.vocab = torchtext.vocab.Vocab(Counter([str(int(onet_code/1000000)*1000000) for onet_code in self.onet_seen]), specials=[])
        HASH.vocab = dill.load(open('./Data/data_new/hash_id_vocab', 'rb'))
        
        #create train iter
        max_batches = 1000
        examples = []
        for batch_ind in tqdm(range(max_batches)):
            examples.extend(self.data_gen_softmax(batch_ind, DOC_IND, HASH, nestedfield, ONET_CODE, ONET_CODE_l3, 
                                             ONET_CODE_l2, ONET_CODE_l1, category='train'))
    
        train_dataset = data.Dataset(examples, [('doc_ind', DOC_IND), ('hash', HASH), ('job_desc', nestedfield), 
            ('onet_code', ONET_CODE), ('onet_code_l3', ONET_CODE_l3), ('onet_code_l2', ONET_CODE_l2), ('onet_code_l1', ONET_CODE_l1), 
            ('onet_desc', nestedfield)])
        
        # train_dataset, _ = train_dataset.split(split_ratio=[0.01, 0.99], stratified=True, strata_field='onet_code', random_state=random.getstate())
        train_iter = data.BucketIterator(train_dataset,
                                          batch_size = self.batch_size,
                                        batch_size_fn = self.batch_size_fn,
                                          sort_key = lambda x: len(x.job_desc),
                                          repeat = False,
                                          shuffle = True,
                                          sort_within_batch = True)
            
        examples = []
        for batch_ind in tqdm(range(max_batches)):
            examples.extend(self.data_gen_softmax(batch_ind, DOC_IND, HASH, nestedfield, ONET_CODE, ONET_CODE_l3, 
                                             ONET_CODE_l2, ONET_CODE_l1, category='test'))
    
        test_dataset = data.Dataset(examples,[('doc_ind', DOC_IND), ('hash', HASH), ('job_desc', nestedfield), 
            ('onet_code', ONET_CODE), ('onet_code_l3', ONET_CODE_l3), ('onet_code_l2', ONET_CODE_l2), ('onet_code_l1', ONET_CODE_l1), 
            ('onet_desc', nestedfield)])
        

        # test_dataset, _ = test_dataset.split(split_ratio=[0.01, 0.99], stratified=True, strata_field='onet_code', random_state=random.getstate())
        test_iter = data.BucketIterator(test_dataset,
                                          batch_size = self.batch_size,
                                        batch_size_fn = self.batch_size_fn,
                                          sort_key = lambda x: len(x.job_desc),
                                          repeat = False,
                                          shuffle = False,
                                          sort_within_batch = False)

        return train_iter, test_iter, unseen_iter, TEXT, HASH, ONET_CODE, ONET_CODE_l3, ONET_CODE_l2, ONET_CODE_l1
        

    def process_onet_info(self):
        """
        Select which sections (such as Tasks, Education, Skills) will go into onet description used for training
        """
        for onet_code in self.onet_information_dict.keys():
            if(self.onet_information_dict[onet_code]['Tasks'] is None):
                continue
            contents = self.onet_information_dict[onet_code]['Tasks']
            contents = [(elem[0], onet_desc_tokenizer(standard_preprocess(elem[1]), remove_stopwords=False)) for elem in contents]
            self.onet_information_dict[onet_code]['Tasks'] = contents

        for onet_code in self.onet_information_dict.keys():
            if(self.onet_information_dict[onet_code]['OnetTitle'] is None):
                continue
            onet_title = self.onet_information_dict[onet_code]['OnetTitle']
            onet_title = onet_desc_tokenizer(standard_preprocess(onet_title), remove_stopwords=False)
            self.onet_information_dict[onet_code]['OnetTitle'] = onet_title

        print('Prepared Onet Info\n')


if __name__ == '__main__':
    # get_job_title_dict()
    df_sampled = pickle.load(open('./Data/data_new/df_sampled.pkl', 'rb'))
    job_title_dict = pickle.load(open('./Data/data_new/job_title_dict', 'rb'))
    for key in job_title_dict.keys():
        print(df_sampled[df_sampled['hash']==key].title)
        print(job_title_dict[key])
        print('\n\n')
        # break