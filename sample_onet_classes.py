#!/usr/bin/env python
# coding: utf-8

# In[109]:


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

from progiter import ProgIter as tqdm

from math import ceil
from sklearn.model_selection import train_test_split
import os
import argparse
import random


# In[110]:


def convert_code_to_int(onet_code):
    onet_code = onet_code.replace('-','')
    onet_code = onet_code.replace('.','')
    return int(onet_code)


# In[146]:


def onet_taxononomy(onet_subset=None):
    df_onet_distribution = pd.read_csv('./Checks/check_sampled.csv', header=None)
    df_onet_distribution.columns = ['onet_occupation_code', 'freq']
    #store each level in the onet hierarchy as a list
    onet_code_description_dict = dict()
    onet_codes_hierarchy = dict()
    onet_codes_children = dict()
    onet_codes_neighbors = dict()
    onet_codes_dict = dict()
    
    #replace the last level with seen classes
    if(onet_subset is None):
        onet_codes = df_onet_distribution['onet_occupation_code'].values
        # remove '.00' and '-' in onet codes, convert them to integers and sort them 
        onet_codes_l4 = onet_codes.copy()
        for i in range(len(onet_codes_l4)):
            onet_codes_l4[i] = onet_codes_l4[i].replace('-','')
            onet_codes_l4[i] = onet_codes_l4[i].replace('.','')
        onet_codes_l4 = np.sort(np.array(onet_codes_l4, dtype=int))

    else:
        if(type(onet_subset[0]) != np.int64):
            onet_codes_l4 = np.sort(np.array([convert_code_to_int(onet_code) for onet_code in onet_subset], dtype=int))
        else:
            onet_codes_l4 = np.sort(np.array([onet_code for onet_code in onet_subset], dtype=int))
    
    onet_codes_l3 = set()
    onet_codes_l2 = set()
    onet_codes_l1 = set()
    for onet_code in onet_codes_l4:
        onet_codes_l3.add(int(onet_code/1000)*1000)
        onet_codes_l2.add(int(onet_code/100000)*100000)
        onet_codes_l1.add(int(onet_code/1000000)*1000000)

    onet_codes_l3 = np.sort(np.array(list(onet_codes_l3)))
    onet_codes_l2 = np.sort(np.array(list(onet_codes_l2)))
    onet_codes_l1 = np.sort(np.array(list(onet_codes_l1)))
    
    onet_codes_dict[4] = onet_codes_l4
    onet_codes_dict[3] = onet_codes_l3
    onet_codes_dict[2] = onet_codes_l2
    onet_codes_dict[1] = onet_codes_l1
    for i in range(1,5):
        for onet_code in onet_codes_dict[i]:
            onet_codes_children[onet_code] = []
            onet_codes_neighbors[onet_code] = []
            onet_code_description_dict[onet_code] = ''
    #-1 is root
    onet_codes_children[-1] = []
    onet_codes_neighbors[-1] = []
    for i in range(4,0,-1):
        for onet_code in onet_codes_dict[i]:
            if(i==4):
                parent = int(onet_code/1000)*1000
            elif(i==3):
                parent = int(onet_code/100000)*100000
            elif(i==2):
                parent = int(onet_code/1000000)*1000000
            else:
                parent = -1
            onet_codes_children[parent].append(onet_code)
    # print(onet_codes_children)
    negatives = 1000
    for i in range(4,0,-1):
        for onet_code in onet_codes_dict[i]:
            if(i==4):
                parent = int(onet_code/1000)*1000
            elif(i==3):
                parent = int(onet_code/100000)*100000
            elif(i==2):
                parent = int(onet_code/1000000)*1000000
            else:
                parent = -1
            siblings = np.array(onet_codes_children[parent])
    #         print(np.argwhere(siblings==onet_code), onet_code)
            pos = np.argwhere(siblings==onet_code)[0][0]
            k = int(negatives/2)
            overflow_r = max(0,(pos+k)-(len(siblings)-1))
            overflow_l = max(0, -(pos-k))
            r = min(len(siblings)-1, pos+k+overflow_l)
            l = max(0, pos-k-overflow_r)
            onet_codes_neighbors[onet_code] = np.array(np.sort(list(set(siblings[l:r+1].tolist())-{onet_code,}))).tolist()
    return (onet_codes_dict, onet_codes_children, onet_codes_neighbors)

# In[618]:


def filter_codes(onet_codes):
    if(onet_codes == []):
        return []
    df_onet_descriptions = pd.read_csv('./Data/data_new/2010_Occupations.csv')
    all_others = df_onet_descriptions['O*NET-SOC 2010 Title'].map(lambda x: True if x.find('All Other')>0 else False)
    all_others_codes = df_onet_descriptions.ix[all_others]['O*NET-SOC 2010 Code'].values.tolist()
    if(type(onet_codes[0]) != str):
        all_others_codes = [convert_code_to_int(onet_code) for onet_code in all_others_codes]
#     print(onet_codes, all_others_codes)
    filtered = set(onet_codes) - set(all_others_codes)
    return list(filtered)

def sample_unseen_onet(onet_seen, parent_level=2):
    onet_unseen = set()
    onet_codes_dict, onet_codes_children, onet_codes_neighbors = onet_taxononomy()
    
    if(parent_level not in [1,2,3]):
        print('Parent level must be in [1,2,3]')
        return []
    onet_unseen = []
    for parent in onet_codes_dict[parent_level]:
        children = onet_codes_children[parent]
        cur_level = parent_level+1
        while(cur_level < 4):
            onet_codes = []
            for onet_code in children:
                onet_codes.extend(onet_codes_children[onet_code])
            children = onet_codes
            cur_level += 1
        if(len(set(onet_seen).intersection(set(children))) != 0):
            onet_unseen.extend(list(set(children) - set(onet_seen)))
    return filter_codes(onet_unseen)

def sample_onet(seen_parent_level=2, max_classes=10, min_freq=1500, unseen=True, unseen_parent_level=2):
    df_onet_distribution = pd.read_csv('./Checks/check_sampled.csv', header=None)
    df_onet_distribution.columns = ['onet_occupation_code', 'freq']
    onet_subset = df_onet_distribution.loc[df_onet_distribution['freq']>=min_freq,:]['onet_occupation_code'].values.tolist()
    onet_incomplete_info = pickle.load(open('./Data/data_new/onet_issues','rb'))
    onet_subset = filter_codes(onet_subset)
    onet_codes_dict, onet_codes_children, onet_codes_neighbors = onet_taxononomy(onet_subset)
    onet_unseen = []
    while(True):
        if(seen_parent_level not in [0,1,2,3]):
            print('Parent level must be in [0,1,2,3]')
            return ([], [], [])
        elif(seen_parent_level != 0):
            parent = random.choice(onet_codes_dict[seen_parent_level])
        else:
            parent = -1
        children = onet_codes_children[parent]
        cur_level = seen_parent_level+1
        while(cur_level < 4):
            onet_codes = []
            for onet_code in children:
                onet_codes.extend(onet_codes_children[onet_code])
            children = onet_codes
            cur_level += 1
        if(unseen):
            onet_unseen = sample_unseen_onet(children, unseen_parent_level)
            onet_unseen = [onet_code for onet_code in onet_unseen if not onet_code in onet_incomplete_info]

        if(len(children) > 2 and ((not unseen) or onet_unseen != [])):
            break
    children = [onet_code for onet_code in children if not onet_code in onet_incomplete_info]
    if(len(children) > max_classes):
        children = random.sample(children, max_classes)
    
    return (onet_subset, children, onet_unseen)


# In[846]:


if __name__ == "__main__":
    _,onet_seen, onet_unseen = sample_onet(seen_parent_level=3, unseen=True, unseen_parent_level=3)
    print(onet_seen, onet_unseen)


# In[ ]:




