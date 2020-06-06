import requests
from bs4 import BeautifulSoup
from vocab import convert_code_to_int
import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize
from process_raw_batches import standard_preprocess
from vocab import onet_desc_tokenizer
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
import gc

def extract_onet_info():
    """
    Code to scrape O*NET website
    """
    #Need to include WorkContenxt
    categories = ['OnetTitle', 'ShortDescription', 'Tasks', 'TechnologySkills', 'Knowledge', 'Skills', 'Abilities', 'WorkActivities', 'DetailedWorkActivities',
                 'Interests', 'WorkStyles', 'WorkValues', 'JobTitles']

    df_onet_descriptions = pd.read_csv('./Data/data_new/2010_Occupations.csv')
    onet_information_dict = dict()
    for i in range(len(df_onet_descriptions)):
        onet_code = df_onet_descriptions.loc[i, 'O*NET-SOC 2010 Code']
        print(onet_code)
        page = requests.get(f'https://www.onetonline.org/link/details/{onet_code}')
        soup = BeautifulSoup(page.text, 'html.parser')
        info = dict()
        for category in categories:
    #         print('\n', category, '\n')
            if(category == 'OnetTitle'):
                info[category] = df_onet_descriptions.loc[i, 'O*NET-SOC 2010 Title'].strip()
            elif(category == 'ShortDescription'):
                info[category] = df_onet_descriptions.loc[i, 'O*NET-SOC 2010 Description'].strip()
            elif(category == 'JobTitles'):
                titles = None
                b_el = soup.find('b',text='Sample of reported job titles:')
                if(not b_el is None):
                    titles = b_el.next_sibling.string.split(',')
                    titles = [str(title).strip() for title in titles]
                info[category] = titles
            else:
                task_sec = soup.find("div", {"id": f"wrapper_{category}"})
                if(task_sec is None):
                    info[category] = None
                    continue
                tasks_list = task_sec.find_all('li')
                if(category != 'TechnologySkills'):
                    tasks_contents = []
                    tabulka = task_sec.find("table")
                    if(tabulka != None):
                        for row in tabulka.findAll('tr'):
                            col = row.findAll('td')
                            if(col == [] or col[0].b is None):
                                continue
                            elem1 = col[0].b.string.strip()
                            if(category=='Tasks'):
                                elem2 = col[2].contents[0].contents[0]
                            else:
                                elem2 = col[1].b.string.strip()
                            tasks_contents.append((str(elem1), str(elem2)))
                    else:
                        tasks_contents = [task.contents[0] for task in tasks_list]
                        if(not (category=='Tasks' or category=='DetailedWorkActivities')):
                            tasks_contents = [task.contents[0] for task in tasks_contents]
                        tasks_contents = [(None, str(task).strip('\n').strip()) for task in tasks_contents]
                    info[category] = tasks_contents
                else:
                    full_skills = []
                    tasks_contents = [task.contents for task in tasks_list]
                    for elems in tasks_contents:
                        for elem in elems:
                            if((not (elem.string is None))and elem.string.strip('\n').strip() != '' and '(see all' not in elem.string.strip('\n').strip()):
                                skills = elem.string.strip('\n').strip()
                                skills = skills.split(';')
                                skills = [str(skill).strip('—').strip() for skill in skills if skill.strip('—').strip() != '']
                                full_skills.extend(skills)

                    info[category] = full_skills

    #         print(info[category])
        onet_information_dict[convert_code_to_int(onet_code)] = info
    with open('./Data/data_new/onet_information_dict', 'wb') as handle:
        pickle.dump(onet_information_dict, handle)


def extract_onet_categorical_fields():
    """
    Organise scraped information on each onet code.
    Divide information into sentions
    """
    onet_categorical_fields_dict = dict()

    categories = ['Abilities', 'Knowledge', 'Skills', 'Work_Activities', 'Work_Context', 'Interests', 
                  'Work_Styles', 'Work_Values']
    #category='Work_Styles'
    for category in categories:
        print(category)
        page = requests.get(f'https://www.onetonline.org/find/descriptor/browse/{category}/')
        soup = BeautifulSoup(page.text, 'html.parser')

        # elements_sec = soup.find("div", {"id": f"content"})
        elements = dict()
        elements_sec = soup.find("div", {"class": f"reportdesc"})
        while(True):
        #     print(elements_sec.find_next_sibling("div"))
            elements_sec = elements_sec.find_next_sibling("div")
            if(elements_sec is None):
                break
            parent_elem = str(elements_sec.find_all('a')[-1].string)
            link = elements_sec.find_all('a')[-1]['href']
            definition = str(elements_sec.contents[-1])
            definition = definition.strip().lstrip('—')
            # elements[parent_elem] = definition
    #         print('https://www.onetonline.org/'+f'{link}')
            if(category not in ['Knowledge', 'Work_Values', 'Interests', 'Work_Styles']):
                child_page = requests.get('https://www.onetonline.org/'+f'{link}')
                child_soup = BeautifulSoup(child_page.text, 'html.parser')
                child_elements_sec = child_soup.find("div", {"class": f"reportdesc"})
                while(True):
                    child_elements_sec = child_elements_sec.find_next_sibling("div")
                    if(child_elements_sec is None):
                        break
                    child_elem = str(child_elements_sec.find_all('a')[-1].string)
                    child_definition = str(child_elements_sec.contents[-1])
                    child_definition = child_definition.strip().lstrip('—')
                    elements[child_elem] = child_definition
            else:
                elements[parent_elem] = definition
        
        onet_categorical_fields_dict[category] = elements
    with open('./Data/data_new/onet_categorical_fields_dict', 'wb') as handle:
        pickle.dump(onet_categorical_fields_dict, handle)


def extract_onet_features():
     """
    Use universal sentence encoder to convert ONET information in to numerical embeddings
    Use embeddings for visualizing onet codes.
    """
    categories = ['OnetTitle', 'Tasks', 'Knowledge', 'Skills', 'Abilities', 'WorkActivities', 'DetailedWorkActivities',
             'Interests', 'WorkStyles', 'WorkValues', 'JobTitles']

    onet_information_dict = pickle.load(open('./Data/data_new/onet_information_dict', 'rb'))
    onet_categorical_fields_USE_features_dict = pickle.load(open('./Data/data_new/onet_categorical_fields_USE_features_dict', 'rb'))
    onet_feature_dict = dict()
    onet_issue = []
    g = tf.Graph()
    with g.as_default():
      text_input = tf.placeholder(dtype=tf.string, shape=[None])
      embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
      my_result = embed(text_input)
      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    tf.logging.set_verbosity(tf.logging.ERROR)
    # graph = tf.Graph()

    with tf.Session(graph=g) as session:
        session.run(init_op)       
        for onet_code in onet_information_dict.keys():
            print(onet_code, '\n')
            flag = 0
            feature_list = []
            feature_dict = dict()
            for category in categories:
                if(onet_information_dict[onet_code][category] is None):
                    print('Feature could not be computed')
                    flag = 1
                    feature_dict[category] = None
                    continue    
                if(category == 'OnetTitle'):
                    # feature = session.run(embed([onet_information_dict[onet_code][category],]))
                    feature = session.run(my_result, feed_dict={text_input:[onet_information_dict[onet_code][category],]})
                    # feature = np.random.randn(1,512)
                    feature_list.append(feature[0].tolist()) 
                elif(category == 'JobTitles'):
                    # feature = session.run(embed(onet_information_dict[onet_code][category])).astype(np.float)
                    feature = session.run(my_result, feed_dict={text_input:onet_information_dict[onet_code][category]}).astype(np.float)
                    # feature = np.random.randn(30,512)
                    # feature = np.average(feature, axis=0)
                    feature_list.append( feature.tolist())
                elif(category == 'Tasks' or category=='DetailedWorkActivities'):
                    sentences = [elem[1] for elem in onet_information_dict[onet_code][category]]
                    weights = [elem[0] for elem in onet_information_dict[onet_code][category]]
                    # feature = session.run(embed(sentences)).astype(np.float)
                    feature = session.run(my_result, feed_dict={text_input:sentences}).astype(np.float)
                    # feature = np.random.randn(len(weights),512)
                    if(not weights[0] is None):
                        weights = np.array(weights).astype(np.float)/100
                        weights = weights.tolist()
                        # feature = np.average(feature, axis=0, weights=weights)
                    else:
                        if(category == 'Tasks'):
                            flag = 2
                        # feature = np.average(feature, axis=0)
                        
                    if(category == 'Tasks'):
                        feature_list.append((weights,  feature.tolist()))
                    else:
                        feature_list.append( feature.tolist())
                else:
                    fields = [elem[1] for elem in onet_information_dict[onet_code][category]]
                    weights = [int(elem[0]) for elem in onet_information_dict[onet_code][category]]
                    if(fields == []):
                        flag = 3
                        feature_dict[category] = None
                        continue
                        # fields = [elem[1] for elem in onet_information_dict[onet_code][category]]
                        # weights = [int(elem[0]) for elem in onet_information_dict[onet_code][category]]
                    feature = [onet_categorical_fields_USE_features_dict[re.sub(r'(?<=[a-z])(?=[A-Z])',r'_', category)][field] for field in fields]
                    feature = np.stack(feature, axis=0)
                    weights = np.array(weights).astype(np.float)/100
                    # feature = np.average(feature, axis=0, weights=weights)
                    feature_list.append((weights.tolist(), feature.tolist()))
                feature_dict[category] = feature_list[-1]
            if(flag):
                # print(flag)
                onet_feature_dict[onet_code] = None
                onet_issue.append(onet_code)
            else:
                onet_feature_dict[onet_code] = feature_dict
            # del feature, feature_list
    print(len(onet_issue), onet_issue)
    with open('./Data/data_new/onet_issue', 'wb') as handle:
        pickle.dump(onet_issue, handle)  
    with open('./Data/data_new/onet_feature_dict', 'wb') as handle:
        pickle.dump(onet_feature_dict, handle)     


if __name__=="__main__":
    # extract_onet_info()
    # extract_onet_categorical_fields()
    extract_USE_features_onet_categorical_fields()
    # extract_onet_features()