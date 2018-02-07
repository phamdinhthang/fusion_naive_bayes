# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:38:39 2018

@author: Pham Dinh Thang
Notes: If probabilities goes toward zero and cause underflow, try the log-sum-exp trick
"""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from NaiveBayes_Distribution import Gaussian_Distribution, Multinomial_Distribution

class Fusion_NaiveBayes(object):
    def __init__(self):
        self.model = None #A dict object {label:{'feature':distribution,...},...}
        self.data_len = 0 #Number of training sample for the model
        self.classes_ = None #Available class of the model
        self.labels_count = None #Count for each labels in the training data
    
    def fit(self, df_features, df_labels):
        self.data_len = len(df_features)
        
        df_features_grouped = df_features.groupby(df_labels)
        self.model, self.labels_count = {}, {}
        for label,df in df_features_grouped:
            features_distribution = {column:self.build_distribution(df,column) for column in list(df.columns)}
            self.model[label] = features_distribution
            self.labels_count[label] = len(df)
            
        self.classes_ = np.array(list(self.model.keys()))
    
    def partial_fit(self, df_features, df_labels):
        self.data_len += len(df_features)
        
        df_features_grouped = df_features.groupby(df_labels)
        for label,df in df_features_grouped:
            appended_features_distribution = {column:self.build_distribution(df,column) for column in list(df.columns)}
            new_features_distribution = {}
            for column,distribution in self.model.get(label).items():
                distribution.partial_fit(appended_features_distribution.get(column))
                new_features_distribution[column] = distribution
                
            self.model[label] = new_features_distribution
            self.labels_count[label] = self.labels_count.get(label) + len(df)
            
    def build_distribution(self,df,col):
        try:
            if (is_numeric_dtype(df[col])):
                arr = np.array(list(df[col]))
                mean, std = np.mean(arr), np.std(arr)
                return Gaussian_Distribution(mean,std,len(arr))
            if (is_string_dtype(df[col])):
                value_counts = df[col].value_counts()
                categories_count = {x[0]:int(x[1]) for x in value_counts.iteritems()}
                return Multinomial_Distribution(categories_count)
        except:
            print("Invalid value type for numeric or categorical column. Please check value type")
    
    def predict(self,df_features):
        if self.model == None:
            print("No model available. Please fit a model before predict")
            return
        
        rows_proba = self.predict_proba(df_features)
        labels = []
        for row in list(rows_proba):
            label_index = np.argmax(row)
            labels.append(self.classes_[label_index])
            
        return np.array(labels)
    
    def predict_proba(self,df_features):
        if self.model == None:
            print("No model available. Please fit a model before predict")
            return
        
        rows = df_features.to_dict('records')
        rows_proba = []
        for row in rows:
            labels_probs = []
            for label in self.classes_:
                prior = self.calculate_prior(label)
                evidence = self.calculate_evidence(row)
                likelihood = self.calculate_likelihood(row,label)
                label_prob = prior*likelihood/evidence                    
                labels_probs.append(label_prob)
            rows_proba.append(labels_probs)
        return np.array(rows_proba)
    
    def predict_log_proba(self,df_features):
        return np.log(self.predict_proba(df_features))
    
    def calculate_likelihood(self,row_dict,label):
        likelihood = 1
        features_distribution = self.model.get(label)
        for feature,val in row_dict.items():
            distribution = features_distribution.get(feature)
            feature_prob = distribution.get_probability(val)
            likelihood *= feature_prob
        return likelihood
    
    def calculate_evidence(self,row_dict):
        evidence = 0
        for label in list(self.classes_):
            prior = self.calculate_prior(label)
            likelihood = self.calculate_likelihood(row_dict,label)
            evidence += prior*likelihood
        return evidence
    
    def calculate_prior(self,label):
        return self.labels_count.get(label)/self.data_len

    def __str__(self):
        if self.model == None:
            print("No model available")
            return ""
        desc = ""
        for label,val in self.model.items():
            desc += "------Distribution over label = "+str(label)+" ---------\n"
            desc += "P(label="+str(label)+")="+str(val[0])
            for feature, distribution in val[1].items():
                desc += "Feature = "+str(feature)+", distribution = "+str(distribution)
            
        return desc