#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import pickle
import re
import string

from progiter import ProgIter as tqdm

from math import ceil
from sklearn.model_selection import train_test_split
import os
import glob
import argparse
import shutil
import random
from collections import defaultdict

from sample_onet_classes import sample_onet, convert_code_to_int

from distutils.dir_util import copy_tree

'''
For testing purposes, the code that follows allows for sampling data belonging to set of onet codes from the available training data.
'''


def data_gen_softmax():
    max_batches = 10000
    for category in ['train', 'test', 'unseen']:
        for batch_ind in tqdm(range(max_batches)):
            try:
                data = pickle.load(open(f'./Data/data_new/sentenced_job_desc_sample/{category}/data_{batch_ind}.pkl', 'rb'))
            except:
                break
            for i, (hash_id, job_desc) in enumerate(data):
                yield (hash_id, job_desc)


def train_test_unseen(onet_seen, onet_unseen, seen_sample=1500, test_size = 500, unseen_perclass_size=100):
    '''
    #split hash_ids into train test and unseen
    '''
    global batch_size, out_dir
    hash_ids_dict = dict()
    df_onet = pickle.load(open('./Data/data_new/df_sampled_final.pkl', 'rb')) #uncomment
    seen = df_onet['onet_occupation_code'].map(lambda x: convert_code_to_int(x)).isin(onet_seen)
    df_seen = df_onet.ix[seen]
    #subsample seen indices
    df_seen_subsampled = df_seen.groupby('onet_occupation_code').apply(lambda x: x.sample(min(seen_sample, len(x)), random_state=0)).reset_index(drop=True)
    df_seen_subsampled = df_seen_subsampled.reset_index()
    df_train, df_test = train_test_split(df_seen_subsampled, test_size=test_size*len(onet_seen), 
                                         random_state=0, shuffle=True, stratify=df_seen_subsampled['onet_occupation_code'].values)
    hash_ids_dict['train'] = df_train['hash'].values.tolist()
    hash_ids_dict['test'] = df_test['hash'].values.tolist()
    unseen = df_onet['onet_occupation_code'].map(lambda x: convert_code_to_int(x)).isin(onet_unseen)
    df_unseen = df_onet.ix[unseen]
    df_unseen_subsampled = df_unseen.groupby('onet_occupation_code').apply(lambda x: x.sample(min(unseen_perclass_size, len(x)), random_state=0)).reset_index(drop=True)
    df_unseen_subsampled = df_unseen_subsampled.reset_index()
    if(onet_unseen != []):
        hash_ids_dict['unseen'] = df_unseen_subsampled['hash'].values.tolist()
    else:
        hash_ids_dict['unseen'] = []
        
    for category in ['train', 'test', 'unseen']:
        temp_dict = defaultdict(int)
        for hash_id in hash_ids_dict[category]:
            temp_dict[hash_id]+= 1
        hash_ids_dict[category] = temp_dict
    
    #for each category create documents list and hash_ids list
    sampled_batch_dict = {'train':[], 'test':[], 'unseen':[]}
    batch_ind_dict = {'train':-1, 'test':-1, 'unseen':-1}
    for hash_id, job_desc in data_gen_softmax():
        for category in ['train', 'test', 'unseen']:
            if(hash_ids_dict[category][hash_id]):
                sampled_batch_dict[category].append((hash_id, job_desc))
                break
    
        for category in ['train', 'test', 'unseen']:
            if(len(sampled_batch_dict[category]) >= batch_size):
                batch_ind_dict[category] += 1
                print(category, batch_ind_dict[category], len(sampled_batch_dict[category]))
                with open(f'./Data/data_new/{out_dir}/{category}/data_{batch_ind_dict[category]}.pkl', 'wb') as handle:
                    pickle.dump(sampled_batch_dict[category], handle)
                sampled_batch_dict[category] = []
    
    for category in ['train', 'test', 'unseen']:
        if(len(sampled_batch_dict[category])):
            batch_ind_dict[category] += 1
            print(batch_ind_dict[category], len(sampled_batch_dict[category]))
            with open(f'./Data/data_new/{out_dir}/{category}/data_{batch_ind_dict[category]}.pkl', 'wb') as handle:
                pickle.dump(sampled_batch_dict[category], handle)
            sampled_batch_dict[category] = []   

    print('Completed')


def shuffle_batches(batch_size=40000):#batch size on disk
    max_batches = 1000
    global out_dir
    if(not os.path.exists(f'./Data/data_new/{out_dir}_shuffled')):
        os.makedirs(f'./Data/data_new/{out_dir}_shuffled/train')
        os.makedirs(f'./Data/data_new/{out_dir}_shuffled/test')
        os.makedirs(f'./Data/data_new/{out_dir}_shuffled/unseen')
    for category in ['train', ]:
        keys = []
        for batch_ind in range(max_batches):
            try:
                data = pickle.load(open(f'./Data/data_new/{out_dir}/{category}/data_{batch_ind}.pkl', 'rb'))
            except:
                break
            for i in range(len(data)):
                keys.append((batch_ind, i))
                
        if(keys == []):
            continue
            
        random.shuffle(keys)
        for shuffled_batch_ind, j in tqdm(enumerate(range(0, len(keys), batch_size))):
            batch_keys = keys[j:min(j+batch_size, len(keys))]
            batch_keys.sort()
            batch_ind = -1
            new_data = []
            for key in batch_keys:
                if(batch_ind != key[0]):
                    batch_ind = key[0]
                    data = pickle.load(open(f'./Data/data_new/{out_dir}/{category}/data_{batch_ind}.pkl', 'rb'))
                new_data.append(data[key[1]])
            with open(f'./Data/data_new/{out_dir}_shuffled/{category}/data_{shuffled_batch_ind}.pkl', 'wb') as handle:
                pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    copy_tree(f'./Data/data_new/{out_dir}/test/', f'./Data/data_new/{out_dir}_shuffled/test/')
    copy_tree(f'./Data/data_new/{out_dir}/unseen/', f'./Data/data_new/{out_dir}_shuffled/unseen/')


global batch_size, out_dir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help='batch_size', default=40000)
    parser.add_argument("--out_dir", type=str, help='Output directory containing raw batches', default='sentenced_job_desc_sample_1')
    parser.add_argument("--seen_sample", type=int, help='train+test size of data per onet class', default=1500)
    parser.add_argument("--test_size", type=int, help='test size of data per onet class', default=500)
    parser.add_argument("--seen_parent_level", type=int, help='level at which the seen onet codes have same parent', default=3)
    parser.add_argument("--max_classes", type=int, help='desired seen classes to train model', default=10)
    parser.add_argument("--unseen", action='store_true', help='Add unseen classes')
    parser.add_argument("--unseen_parent_level", type=int, help='level at which the unseen onet codes have same parent', default=2)
    args = parser.parse_args()
    batch_size = args.batch_size
    out_dir = args.out_dir

    if(os.path.exists(f'./Data/data_new/{out_dir}/train/')):
        shutil.rmtree(f'./Data/data_new/{out_dir}/train/')
    if(os.path.exists(f'./Data/data_new/{out_dir}/test/')):
        shutil.rmtree(f'./Data/data_new/{out_dir}/test/')
    if(os.path.exists(f'./Data/data_new/{out_dir}/unseen/')):
        shutil.rmtree(f'./Data/data_new/{out_dir}/unseen/')

    os.makedirs(f'./Data/data_new/{out_dir}/train/')
    os.makedirs(f'./Data/data_new/{out_dir}/test/')
    os.makedirs(f'./Data/data_new/{out_dir}/unseen/')
    onet_subset, onet_seen, onet_unseen = sample_onet(seen_parent_level=args.seen_parent_level, 
                                                      max_classes=args.max_classes, 
                                                      min_freq=args.seen_sample, 
                                                      unseen=args.unseen, 
                                                      unseen_parent_level=args.unseen_parent_level)
    # onet_subset, onet_seen, onet_unseen = pickle.load(open(f'./Data/data_new/onet_sampled', 'rb')) 
    print(onet_seen, '\n', len(onet_seen))
    print(onet_unseen, '\n', len(onet_unseen))
    with open(f'./Data/data_new/{out_dir}/onet_sampled', 'wb') as handle:
        pickle.dump((onet_subset, onet_seen, onet_unseen), handle)
    train_test_unseen(onet_seen, onet_unseen, seen_sample=args.seen_sample, test_size = args.test_size)
    shuffle_batches(batch_size)

# In[ ]:




