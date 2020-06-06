#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import re
import string

from progiter import ProgIter as tqdm

from math import ceil
from sklearn.model_selection import train_test_split
import os
import argparse
import shutil

# #prepocess text
# 
# 1. covert to lawer case
# 2. remove numbers
# 3. removing punctuations, accent marks and other diacritics
# 4. remove extra white spaces
# 5. senetence tokenize
# 6. words Tokenise 
# 7. Stem words - did not perform
# 8. lemmatise words
# 9. removing stop words, sparse terms, and particular words
# 
# In[7]:


def process_job_descriptions():
    job_desc_dict = pickle.load(open('./Data/data_new/job_desc_final.pickle', 'rb'))
    hash_ids = pickle.load(open('./Data/data_new/hash_id_common.pickle', 'rb'))
    job_desc_processed_dict = dict()
    for hash_id in tqdm(hash_ids):
        job_desc = job_desc_dict[hash_id]
        #if small letter is immediately followed by capital letter introduce space
        job_desc = re.sub(r'(?<=[a-z])(?=[A-Z])',r'. ', job_desc)
        #remove numbers
        job_desc = re.sub("\d+", " ", job_desc)
        #remove URLS
        job_desc = re.sub(r"http\S+", " ", job_desc)
        job_desc = re.sub(r"www\S+", " ", job_desc)
        #replace bullets with fullstops
        job_desc = re.sub(r"•", " ", job_desc)
        #replace double line break with fullstop
        job_desc = job_desc.replace("\n\n", ". ")
        #replace double fullstop with single
        job_desc = job_desc.replace("..", ". ")
        job_desc_processed_dict[hash_id] = job_desc
    with open('./Data/data_new/job_desc_processed_dict', 'wb') as handle:
        pickle.dump(job_desc_processed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_input_into_batches(hash_ids, documents, category='train'):
    '''
    #split input into batches and store
    '''

    global batch_size, in_dir
    for batch_k, batch_i in enumerate(range(0, len(documents), batch_size)):
        print(batch_k)
        document_batch = documents[batch_i:min(batch_i+batch_size, len(documents))]
        hash_batch = hash_ids[batch_i:min(batch_i+batch_size, len(documents))]
        with open(f'./Data/data_new/{in_dir}/{category}/batch_{batch_k}', 'wb') as handle:
            pickle.dump((hash_batch, document_batch), handle)


def train_test_unseen(test_ratio = 0.1):
    '''
    #split hash_ids into train test and unseen
    '''

    if(not os.path.exists('./Data/data_new/job_desc_processed_dict')):
        print('Text preprocessing started')
        process_job_descriptions()
    job_desc_dict = pickle.load(open('./Data/data_new/job_desc_processed_dict', 'rb'))
    hash_ids_dict = dict()
    documents_dict = dict()
    for category in ['train', 'test', 'unseen']:
        documents_dict[category] = []
    #read onet dataframe
    df_onet_distribution = pd.read_csv('./Checks/check_sampled.csv', header=None)
    df_onet_distribution.columns = ['onet_occupation_code', 'freq']
    onet_seen = df_onet_distribution.loc[df_onet_distribution['freq']>=20000,:]['onet_occupation_code'].values.tolist()
    onet_unseen = df_onet_distribution.loc[df_onet_distribution['freq']<20000,:]['onet_occupation_code'].values.tolist()
    df_onet = pickle.load(open('./Data/data_new/df_sampled_final.pkl', 'rb')) #uncomment
    seen = df_onet['onet_occupation_code'].isin(onet_seen)
    df_seen = df_onet.ix[seen]
    
    df_train, df_test = train_test_split(df_seen, test_size=test_ratio, 
                                         random_state=0, shuffle=True, stratify=df_seen['onet_occupation_code'].values)
    hash_ids_dict['train'] = df_train['hash'].values.tolist()
    hash_ids_dict['test'] = df_test['hash'].values.tolist()
    unseen = df_onet['onet_occupation_code'].isin(onet_unseen)
    df_unseen = df_onet.ix[unseen]
    hash_ids_dict['unseen'] = df_unseen['hash'].values.tolist()
    #for each category create documents list and hash_ids list
    for category in ['train', 'test', 'unseen']:
        for hash_id in hash_ids_dict[category]:
            documents_dict[category].append(job_desc_dict[hash_id])

        split_input_into_batches(hash_ids_dict[category], documents_dict[category], category)

