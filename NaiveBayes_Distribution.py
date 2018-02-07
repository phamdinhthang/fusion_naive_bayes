# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:56:58 2018

@author: Pham Dinh Thang
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
"""
from abc import ABC, abstractmethod
import math

class Distribution(ABC):
    @abstractmethod
    def get_probability(self,value):
        pass
    
    @abstractmethod
    def partial_fit(self,distribution):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

class Gaussian_Distribution(Distribution):
    def __init__(self,mean,std,count):
        self.mean = mean
        self.std = std
        self.count = count
        self.variance = self.std*self.std
    
    def get_probability(self,value):
        if isinstance(value,int) or isinstance(value,float):
            num = math.exp(-1*math.pow(value-self.mean,2)/(2*self.std*self.std))
            deno = math.sqrt(2*math.pi*self.std*self.std)
            if deno==0: return 1
            probs = num/deno
            if probs==0: probs=1
            return probs
        else:
            print("Invalid input value.")
            return 0
    
    def partial_fit(self,distribution):
        if distribution==None or not isinstance(distribution,Gaussian_Distribution):
            print("Invalid distribution to partial fit. Cannot partial fit gaussian distribution")
            return
        
        new_mean = (self.count*self.mean+ distribution.count*distribution.mean)/(self.count+distribution.count)
        d1 = self.mean - new_mean
        d2 = distribution.mean - new_mean
        v1 = self.variance
        v2 = distribution.variance
        n1 = self.count
        n2 = distribution.count
        new_variance = (n1*(v1+d1*d1)+n2*(v2+d2*d2))/(n1+n2)
        
        self.mean = new_mean
        self.variance = new_variance
        self.std = math.sqrt(self.variance)
        self.count = n1+n2
    
    def __str__(self):
        return "Gaussian distribution. Mean = "+str(self.mean)+", std = "+str(self.std)
        
class Multinomial_Distribution(Distribution):
    def __init__(self,categories_count,laplace_smoothing=True):
        self.categories_count = categories_count #Dict of: {"categories":int}
        self.total_counts = sum(self.categories_count.values())
        self.laplace_pseudocount = 0 if laplace_smoothing==False else 1
        self.categories_prob = self.calculate_category_probs()
        
    def calculate_category_probs(self):
        return {key:self.laplace_smoothing_probs(val) for key,val in self.categories_count.items()}
    
    def get_probability(self,value):
        if isinstance(value,str):
            new_category_probs = self.laplace_pseudocount/(self.total_counts + self.laplace_pseudocount * len(self.categories_count.items()))
            probs = self.categories_prob.get(value,new_category_probs)
            if (probs==0): print("Categorical probability =0")
            return probs
        else:
            print("Invalid input value.")
            return 0
    
    def partial_fit(self,distribution):
        if distribution==None or not isinstance(distribution,Multinomial_Distribution):
            print("Invalid distribution to partial fit. Cannot partial fit multinomial distribution")
            return
        
        new_categories_count = {key:val+distribution.categories_count.get(key,0) for key,val in self.categories_count.items()}
        for key,val in distribution.categories_count.items():
            if key not in list(self.categories_count.keys()):
                new_categories_count[key]=val
                
        self.categories_count = new_categories_count
        self.categories_prob = self.calculate_category_probs()
        self.total_counts = sum(self.categories_count.values())
        
    def laplace_smoothing_probs(self,value):
        return (value+self.laplace_pseudocount)/(self.total_counts + self.laplace_pseudocount*len(self.categories_count.items()))
        
    def __str__(self):
        desc = "Multinomial distribution. Categories probabilities: \n"
        for key,val in self.categories_prob.items():
            desc += "\tCategory = "+str(key)+", probability = "+str(val)+"\n"
        return desc

class Parzen_Windows_Distribution(Distribution):
    def get_probability(self,value):
        pass
    
    def partial_fit(self,distribution):
        pass
    
    def __str__(self):
        pass