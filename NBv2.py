#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:27:34 2019

@author: stevenalsheimer
"""
import csv
import numpy as np
import time
start_time = time.time()
def NB(class1_name, class2_name, Training_doc_dir, Parameter_output_file_dir, vocab_file_dir,Test_doc_dir, output_doc_dir ):
    Naive_bayes_training(class1_name,class2_name,Training_doc_dir,Parameter_output_file_dir,vocab_file_dir)
    scoref = score(class1_name,class2_name,Test_doc_dir,Parameter_output_file_dir, output_doc_dir)
    return scoref

def BOW_features_dict(vocab_file):
    lines = open(vocab_file, 'r')
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
def Naive_bayes_training(class1_name,class2_name,Training_doc_dir,Parameter_output_file,vocab_file):
    dicti = BOW_features_dict(vocab_file)
    V = len(dicti)
    fd = open(Parameter_output_file, "w+")
    fd.close()
    with open(Training_doc_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        N_doc = sum(1 for row in readCSV)
    class2_count = 0
    class1_count = 0
    sum_class1 = V*[0]
    sum_class2 = V*[0]
    with open(Training_doc_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        class1_count = 0
        class2_count  = 0
        for row in readCSV:
            if row[0] == class1_name:
                test_list = row
                test_list.remove(class1_name)
                test_list = [int(i) for i in test_list] 
                sum_class1 = [sum_class1[i] + test_list[i] for i in range(len(test_list))] 
                class1_count +=1
            if row[0] == class2_name:
                test_list = row
                test_list.remove(class2_name)
                test_list = [int(i) for i in test_list] 
                sum_class2 = [sum_class2[i] + test_list[i] for i in range(len(test_list))]
                class2_count += 1
    num_words_class1 = np.sum(sum_class1)
    num_words_class2 = np.sum(sum_class2)
    logprior_class1 = np.log2(class1_count/N_doc)
    logprior_class2 = np.log2(class2_count/N_doc)
    with open(Parameter_output_file, 'a') as csvFile:
        writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
        string_wr = class1_name +"logprior"
        string_wr2 = class2_name +"logprior"
        writer.writerow([string_wr,logprior_class1])
        writer.writerow([string_wr2,logprior_class2])
    string_wr3 = class1_name +"loglikelihood"
    string_wr4 = class2_name +"loglikelihood"   
    loglike_list = [string_wr3]
    loglike_list2 = [string_wr4]
    for i in range(len(sum_class1)):
        prob = (sum_class1[i]+1)/(num_words_class1+V)
        loglikelihood = np.log2(prob)
        loglike_list.append(loglikelihood)
    for i in range(len(sum_class2)):
        prob = (sum_class2[i]+1)/(num_words_class2+V)
        loglikelihood = np.log2(prob)
        loglike_list2.append(loglikelihood)
    with open(Parameter_output_file, 'a') as csvFile:
        writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
        writer.writerow(loglike_list)
        writer.writerow(loglike_list2)
    return print("training done")
    
    
        
    
        


#test_vector = [0,1,0,1,0,1,1]
def Naive_bayes_class_predictor(class1_name,class2_name,input_vector,Trained_parameters_doc):
    ###class 1 prob###
    string_wr = class1_name+"logprior"
    string_wr1 = class2_name+"logprior"
    string_wr3 = class1_name +"loglikelihood"
    string_wr4 = class2_name +"loglikelihood" 
    with open(Trained_parameters_doc) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == string_wr:
                class1_sum = float(row[1])
            if row[0] == string_wr1:
                class2_sum = float(row[1])
    for i in range(len(input_vector)):
        if input_vector[i] == 0:
            continue
        if input_vector[i] > 0:
            with open(Trained_parameters_doc) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    if row[0] == string_wr3:
                        class1_sum += input_vector[i]*float(row[i+1])
                    if row[0] == string_wr4:
                        class2_sum += input_vector[i]*float(row[i+1])
    #print(class2_sum,class1_sum)
    if class2_sum > class1_sum:
        return class2_name
    if class2_sum < class1_sum:
        return class1_name
    
def Naive_bayes_class_predictor2(class1_name,class2_name,input_vector,class1_prior,class2_prior,class1_vec, class2_vec):
    ###class 1 prob###
    class1_sum = class1_prior
    class2_sum = class2_prior
    for i in range(len(input_vector)):
        if input_vector[i] == 0:
            continue
        if input_vector[i] > 0:
            class1_sum += input_vector[i]*float(class1_vec[i+1])
            class2_sum += input_vector[i]*float(class2_vec[i+1])
    #print(class2_sum,class1_sum)
    if class2_sum > class1_sum:
        return class2_name
    if class2_sum < class1_sum:
        return class1_name

#boom = Naive_bayes_class_predictor('action','comedy',test_vector,'Parameter_test.txt')
#print(boom)

def score(class1_name,class2_name,Test_doc_dir,Trained_parameters_doc, output_doc_dir):
    countright = 0
    test_doc_num = 0
    string_wr = class1_name+"logprior"
    string_wr1 = class2_name+"logprior"
    string_wr3 = class1_name +"loglikelihood"
    string_wr4 = class2_name +"loglikelihood" 
    with open(Trained_parameters_doc) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == string_wr:
                class1_sum = float(row[1])
            if row[0] == string_wr1:
                class2_sum = float(row[1])
                
    with open(Trained_parameters_doc) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == string_wr3:
                class1_vec = row
            if row[0] == string_wr4:
                class2_vec = row
    fd = open(output_doc_dir, "w+")
    fd.close()
    with open(Test_doc_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        Total_count = sum(1 for row in readCSV)
    with open(Test_doc_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            test_doc_num+=1
            print(test_doc_num)
            actual_class = row[0]
            row.remove(row[0])
            test_list = [int(i) for i in row]
            predicted_class = Naive_bayes_class_predictor2(class1_name,class2_name,test_list, class1_sum, class2_sum, class1_vec, class2_vec)
            with open(output_doc_dir,'a') as csvFile:
               writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
                writer.writerow([predicted_class])
                
            if predicted_class == actual_class:
                countright += 1
    score = countright/Total_count
    with open(output_doc_dir,'a') as csvFile:
        writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
        writer.writerow([score])
    return score
    

#bayes = Naive_bayes_training('pos','neg','megadoc_movie_train_POS.txt','movie-review-BOWPOS.NB.txt','movie-review-HW2/aclImdb/imdb_POS.vocab')
score_ = score('pos','neg','megadoc_movie_test.txt','movie-review-BOW.NB.txt','output_wrong_lines.txt')
#score_ = score('action','comedy','megadoc_test.txt','Parameter_test.txt','Practice/practice.vocab')
print(score_)


print("--- %s seconds ---" % (time.time() - start_time))









