#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:00:56 2019

@author: stevenalsheimer
"""
import os
import csv
import re
import time
import nltk
start_time = time.time()
def preprocessing_POS(dir_files, class1_name, class2_name, output_training_file, output_test_file, vocab_file_dir):
    BOW_training_vector_doc(dir_files,class1_name,class2_name,output_training_file, vocab_file_dir)
    BOW_training_vector_doc(dir_files,class1_name,class2_name,output_test_file, vocab_file_dir)
def BOW_features_dict(vocab_file):
    lines = open(vocab_file, 'r', encoding = 'utf8')
    dict_of_features = {}
    index = 0
    V_list = []
    for line in lines:
        V_list.append(line.strip().split('\n'))
    for i in range(len(V_list)):
        for token in V_list[i]:
            if token not in dict_of_features:
                dict_of_features[token] = index
                index += 1
    return dict_of_features

#print(BOW_features_dict('Practice/practice.vocab'))
#print(BOW_features_dict('movie-review-HW2/aclimdb/imdb.vocab'))

def BOW_training_vector_doc(dir_files,class1_name,class2_name,output_file_dir, vocab_dir):
    class1dir = str(dir_files+"/"+class1_name)
    class2dir = str(dir_files+"/"+class2_name)
    fd = open(output_file_dir, "w+")
    fd.close()
    features_dict = BOW_features_dict(vocab_dir)
    for filename in os.listdir(class1dir):
        dire = str(dir_files+"/"+class1_name+"/"+filename)
        file = open(dire, encoding = 'utf8')
        s = file.read()
        out = re.findall(r"[\w']+|[!?]", s)
        listt = []
        for word in out:
            listt.append(word.lower())
        postag = nltk.pos_tag(listt)
        BOW_vec = (len(features_dict)+1)*[0]
        BOW_vec[0] = class1_name
        for words in postag:
            word = words[0]
            postag = words[1]
            if word in features_dict:
                index = features_dict[word]
                index = index +1
                BOW_vec[index] += 1
            if postag in features_dict:
                index = features_dict[postag]
                index = index+1
                BOW_vec[index] += 1
        with open(output_file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
            writer.writerow(BOW_vec)
            
    for filename in os.listdir(class2dir):
        dire = str(dir_files+"/"+class2_name+"/"+filename)
        file = open(dire, encoding = 'utf8')
        s = file.read()
        out = re.findall(r"[\w']+|[!?]", s)
        listt = []
        for word in out:
            listt.append(word.lower())
        postag = nltk.pos_tag(listt)
        BOW_vec = (len(features_dict)+1)*[0]
        BOW_vec[0] = class2_name
        for words in postag:
            word = words[0]
            postag = words[1]
            if word in features_dict:
                index = features_dict[word]
                index = index +1
                BOW_vec[index] += 1
            if postag in features_dict:
                index = features_dict[postag]
                index = index+1
                BOW_vec[index] += 1
        with open(output_file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
            writer.writerow(BOW_vec)
    return output_file_dir

megadoc_train = BOW_training_vector_doc('movie-review-HW2/aclImdb/train','pos','neg','megadoc_movie_train_POS.txt','movie-review-HW2/aclImdb/imdb_POS.vocab')
#megadoc_train = BOW_training_vector_doc('Practice/Train','action','comedy','megadoc_test.txt','Practice/practice_POS.vocab')
testdoc1 = BOW_training_vector_doc('movie-review-HW2/aclImdb/test','pos','neg','megadoc_movie_test_POS.txt','movie-review-HW2/aclImdb/imdb_POS.vocab')
print("--- %s seconds ---" % (time.time() - start_time))
#import numpy as np
#list_class1 = []
#list_class2 = []
#with open('megadoc_test.txt') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    class1_count = 0
#    class2_count  = 0
#    for row in readCSV:
#        if row[0] == 'action':
#            test_list = row
#            test_list.remove('action')
#            test_list = [int(i) for i in test_list] 
#            list_class1.append(test_list)
#            class1_count +=1
#        if row[0] == 'comedy':
#            test_list = row
#            test_list.remove('comedy')
#            test_list = [int(i) for i in test_list] 
#            list_class2.append(test_list)
#            class2_count += 1
#print(class1_count,class2_count)
#with open('megadoc_test.txt') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    row_count = sum(1 for row in readCSV)
#print(row_count)
#
#sum_class1 = np.sum(list_class1, axis = 0)
#sum_class2 = np.sum(list_class2, axis = 0)
#num_words_class1 = np.sum(sum_class1)
#num_words_class2 = np.sum(sum_class2)
#print(sum_class1,sum_class2, num_words_class1, num_words_class2)





















