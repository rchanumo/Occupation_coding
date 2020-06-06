    #!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import re
import string

import spacy
nlp_w = spacy.load('en',disable=['parser', 'tagger', 'ner'])
nlp_s = spacy.load('en',disable=['tagger', 'ner'])
from nltk.corpus import stopwords
import multiprocessing as mp

from math import ceil
from sklearn.model_selection import train_test_split
import os
import argparse

from itertools import product
from progiter import ProgIter as tqdm
import shutil
import glob
# import pixiedust
import wordninja

from nltk.corpus import wordnet
from Extra_tools.contractions import CONTRACTION_MAP
import unicodedata


def standard_preprocess(noisy_string):
    denoised_string = re.sub(r"http\S+", " ", noisy_string)
    denoised_string = re.sub(r"www\S+", " ", denoised_string)
    denoised_stringdenoised_string = remove_accented_chars(denoised_string)
    denoised_string = expand_contractions(denoised_string)
    denoised_string = re.sub('[^a-zA-Z]', ' ', denoised_string)
    return denoised_string

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def tokenize(args):
    split, category = args
    global stop_words, max_sent_len, max_doc_len, in_dir, out_dir
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    #subset - processed documents
    subset = []
    batch_id = split[0]
    split_id = split[1]
    hash_ids = split[2]
    documents = split[3]
    for doc in nlp_s.pipe(documents, n_threads=1, batch_size=300):
        sentences = [sent.text for sent in doc.sents]
        sentences = [sentence.lower().strip() for sentence in sentences]
        sentences = [re.sub("’", "'", sentence) for sentence in sentences]
        sentences = [remove_accented_chars(sentence) for sentence in sentences]
        sentences = [expand_contractions(sentence) for sentence in sentences]
        sentences = [sentence.translate(translator) for sentence in sentences]
#         sentences = [sentence for sentence in sentences 
#                      if len(set(sentence.split(' ')) - set(stop_words))>4]
#         sentence_tokens = [[token.text for token in nlp_w(sentence) 
#                   if not token.text in stop_words and not (token.is_space or token.is_punct)][:max_sent_len] for sentence in sentences]
        sentence_tokens = [[token.text for token in nlp_w(sentence) 
                  if not (token.is_space or token.is_punct)][:max_sent_len] for sentence in sentences]
        sentence_tokens = [sentence_token for sentence_token in sentence_tokens if len(sentence_token)>4][:max_doc_len]
        subset.append(sentence_tokens)
    data = (hash_ids, subset)
    try:
        with open(f'./Data/data_new/{out_dir}/{category}/data_{batch_id}_{split_id}.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('pickling error')        


def process_batches():
    '''
    perform parallel tokenization of batches stored on disk
    '''
    global stop_words, max_sent_len, max_doc_len, in_dir, out_dir
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    for category in ['train', 'test', 'unseen']:
        for batchfile in sorted(os.listdir(f'./Data/data_new/{in_dir}/{category}/')):
            batch_k = int(batchfile.split('_')[1])
            hash_batch, document_batch = pickle.load(open(f'./Data/data_new/{in_dir}/{category}/{batchfile}', 'rb'))
            print(batch_k, len(hash_batch), len(document_batch))
            splits = []
            split_len = ceil(len(document_batch)/cores)
            for k,i in enumerate(range(0, len(document_batch), split_len)):
                splits.append((batch_k, k, hash_batch[i:min(i+split_len, len(document_batch))], 
                               document_batch[i:min(i+split_len, len(document_batch))]))

            results = pool.map(tokenize, product(splits, [category,]))




def splitword():
    '''
    corrent for words not saperated by space
    '''
    global out_dir
    max_batches = 10000
    temp_dir = './Data/data_new/temp'
    if(not os.path.exists(temp_dir)):
        os.makedirs(f'{temp_dir}/{out_dir}/train')
        os.makedirs(f'{temp_dir}/{out_dir}/test')
        os.makedirs(f'{temp_dir}/{out_dir}/unseen')
    for category in ['train', 'test', 'unseen']:
        for batch_ind in tqdm(range(max_batches)):
            try:
                data = pickle.load(open(f'./Data/data_new/{out_dir}/{category}/data_{batch_ind}.pkl', 'rb'))
            except:
                break
            new_data = []
            for i, (hash_id, job_desc) in tqdm(enumerate(data)):
                new_job_desc = []
                for sentence in job_desc:
                    new_sentence = []
                    for word in sentence:
                        if(wordnet.synsets(word) == [] and not eng_dict.check(word)):
                            split = wordninja.split(word)
                            if(len(split)==0):
                                #rempve
                                pass
                            elif(len(split)==1):
                                new_sentence.append(word)
                            elif(len(split)==2):
                                split = [new_word for new_word in split if wordnet.synsets(new_word) != [] and len(new_word)>2]
                                new_sentence.extend(split)
                            else:
                                pass
#                                 print('********ignored:  ', word, ':', split)
                        else:
                            if(len(word)>1):
                                new_sentence.append(word)
                    if(new_sentence != []):
                        new_job_desc.append(new_sentence)
                if(new_job_desc != []):
                    new_data.append((hash_id, new_job_desc))
            with open(f'{temp_dir}/{out_dir}/{category}/data_{batch_ind}.pkl', 'wb') as handle:
                pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



global stop_words, max_sent_len, max_doc_len, in_dir, out_dir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-max_sent_len", type=int, help='Max words in sentence', default=200)
    parser.add_argument("-max_doc_len", type=int, help='Max sentences in job description', default=200)
    parser.add_argument("-in_dir", type=str, help='Input directory containing raw batches', default='input')
    parser.add_argument("-out_dir", type=str, help='Output directory containing processed batches', default='sentenced_job_desc_sample')
    args = parser.parse_args()
    stop_words = stopwords.words('english')
    max_sent_len = args.max_sent_len
    max_doc_len = args.max_doc_len
    in_dir = args.in_dir
    out_dir = args.out_dir

    if(os.path.exists(f'./Data/data_new/{out_dir}/train/')):
        os.makedirs(f'./Data/data_new/{out_dir}/train/')
    if(os.path.exists(f'./Data/data_new/{out_dir}/test/')):
        os.makedirs(f'./Data/data_new/{out_dir}/test/')
    if(os.path.exists(f'./Data/data_new/{out_dir}/unseen/')):
        os.makedirs(f'./Data/data_new/{out_dir}/unseen/')

    process_batches()
    




