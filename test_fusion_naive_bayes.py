# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:37:31 2018

@author: Pham Dinh Thang

Iris dataset: https://archive.ics.uci.edu/ml/datasets/iris
Census income dataset: https://archive.ics.uci.edu/ml/datasets/census+income
"""

from Fusion_NaiveBayes import Fusion_NaiveBayes
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
import category_encoders as ce
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt

def encode_categorical(df,encoder_name='binary'):
    encoder_dic = {"one_hot":ce.OneHotEncoder(),
                   "feature_hashing":ce.HashingEncoder(n_components=32),
                   "binary":ce.BinaryEncoder(),
                   "ordinal":ce.OrdinalEncoder(),
                   "polynomial":ce.PolynomialEncoder()}
    encoder = encoder_dic.get(encoder_name)
    encoder.fit(df,verbose=1)
    df = encoder.transform(df)
    return df

def train_test_split(df,label_col,ratio=0.1):
    df = df.sample(frac=1).reset_index(drop=True)
    nrows = len(df)
    pivot = int(nrows*(1-ratio))
    df_train = df.loc[:pivot]
    df_test = df.loc[pivot:]
    return df_train.drop(label_col,axis=1),df_test.drop(label_col,axis=1),df_train[label_col],df_test[label_col]

def test_model(df, label_col, model, model_name):
    start = time.time()
    df_train_features, df_test_features, df_train_labels, df_test_labels = train_test_split(df,label_col)
    
    if isinstance(model,BaseEstimator):
        df_train_features = encode_categorical(df_train_features,'feature_hashing').values
        df_test_features = encode_categorical(df_test_features,'feature_hashing').values
        df_train_labels = df_train_labels.values
        df_test_labels = df_test_labels.values   
    
    model.fit(df_train_features, df_train_labels)
    pred = model.predict(df_test_features)
    probs = model.predict_proba(df_test_features)
    
    end = time.time()
    
    if not isinstance(df_test_labels, np.ndarray): df_test_labels = np.array(df_test_labels)
    accuracy = accuracy_score(df_test_labels, pred)
    
    print("-------------",model_name,"-------------")
    print("Accuracy =",accuracy,"\nDuration =",end-start,"(s)")
    print("Model classes =",model.classes_)
    print("Classes probabilities =\n",probs)

def test_partial_fit(df, label_col, model, model_name):
    df_train_features, df_test_features, df_train_labels, df_test_labels = train_test_split(df,label_col)
    df_test_labels_arr = np.array(df_test_labels)
    
    initial_fit_index = int(0.1*len(df_train_features))
    model.fit(df_train_features.iloc[0:initial_fit_index,],df_train_labels[0:initial_fit_index,])
    
    accuracies = [accuracy_score(df_test_labels_arr, model.predict(df_test_features))]
    plot_index =[initial_fit_index-1]
    for i in range(initial_fit_index,initial_fit_index+100):
        partial_df_train_features = df_train_features.iloc[i:i+1,]
        partial_df_train_labels = df_train_labels.iloc[i:i+1,]
        model.partial_fit(partial_df_train_features,partial_df_train_labels)
        accuracies.append(accuracy_score(df_test_labels_arr, model.predict(df_test_features)))
        plot_index.append(i)
    
    try:
        plt.scatter(plot_index,accuracies)
        plt.show()
    except: pass
        
def main(run_param):
    src_path = os.path.abspath(os.path.dirname(__file__))
    
    df_path, label_col = None, None
    if run_param=='iris':
        df_path, label_col = os.path.join(src_path,'iris.csv'), 'species'
    else:
        df_path, label_col = os.path.join(src_path,'census_income.csv'), 'high_income'
    
    df = pd.read_csv(df_path)
    models = {"Fusion Naive Bayes":Fusion_NaiveBayes(),
              "Sklearn GaussianNB":GaussianNB(),
              "Sklearn MultinomialNB":MultinomialNB()}
    
    for model_name,model in models.items():
        test_model(df, label_col, model, model_name)
        
    test_partial_fit(df, label_col,Fusion_NaiveBayes(),'Fusion Naive Bayes')
    
if __name__ == '__main__':
#    main("census-income")
    main(sys.argv[1])