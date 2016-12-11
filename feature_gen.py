# coding: utf-8
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import random as rnd
import sklearn.cluster

class new_feature(object):
    '''can do drop feature, and compute mutual info'''
    
    name = 'new_feature'
    mutual_info = 0
    model_score = 0
    
    def __init__(self, name):
        self.name = name
        
    def drop_feature(self, df):
        df.drop([self.name])
        
    def get_info(self,df,target_f):
        self.mutual_info = sk.metrics.mutual_info_score(df[target_f], df[self.name])
        print(self.name, ' ', self.mutual_info)
        
class linreg_feature(new_feature):
    model = None
    source_feature = ''
    
    def fit_transform(self, df, feature_name, target_name,use_log=False):
        
        #base_features = df.columns.difference([target_name])
        self.name = feature_name+'_lr'
        self.source_feature = feature_name

        chunk1 = []
        chunk2 = []

        # split on two halves
        splitter = sk.cross_validation.ShuffleSplit(n = df.shape[0], n_iter = 1, test_size = 0.5, random_state = 42)
        
        for s1,s2 in splitter:
            chunk1.append(s1)
            chunk2.append(s2)

        chunk1 = chunk1[0]
        chunk2 = chunk2[0]

        # training predictor on each half of the training set to predict the other -> creating new feature
        model1 = sk.linear_model.LinearRegression(n_jobs=-1)
        model2 = sk.linear_model.LinearRegression(n_jobs=-1)      
        model1.fit(df[feature_name].iloc[chunk1].reshape(-1,1), df[target_name].iloc[chunk1])
        model2.fit(df[feature_name].iloc[chunk2].reshape(-1,1), df[target_name].iloc[chunk2])

        # training predictor on all training set to create new (same goal as upper) feature of the test set
        self.model = sk.linear_model.LinearRegression(n_jobs=-1)
        self.model.fit(df[feature_name].reshape(-1,1), df[target_name])
        
        df[self.name] = 0
        if use_log:
        	df[self.name].iloc[chunk2] = np.log2(1+model1.predict(df[feature_name].iloc[chunk2].reshape(-1,1)))
        	df[self.name].iloc[chunk1] = np.log2(1+model2.predict(df[feature_name].iloc[chunk1].reshape(-1,1)))
        else:        
        	df[self.name].iloc[chunk2] = model1.predict(df[feature_name].iloc[chunk2].reshape(-1,1))
        	df[self.name].iloc[chunk1] = model2.predict(df[feature_name].iloc[chunk1].reshape(-1,1))
        
    def transform(self, df, use_log=False):
        if use_log:
        	df[self.name] = np.log2(1+self.model.predict(df[self.source_feature].reshape(-1,1)))
        else:
        	df[self.name] = self.model.predict(df[self.source_feature].reshape(-1,1)) 

class sum_feature(new_feature):
    ''' Creates feature = feature1 + feature2'''
    feature1 = ''
    feature2 = ''
    
    def __init__(self,f1,f2):
        self.name = f1 + '_' + f2 + '_sum'
        self.feature1 = f1
        self.feature2 = f2
    
    def fit_transform(self, df, **args):
        df[self.name] = df[self.feature1] + df[self.feature2]
    
class delta_feature(new_feature):
    ''' Creates feature = feature1 - feature2'''
    feature1 = ''
    feature2 = ''
    
    def __init__(self,f1,f2):
        self.name = f1 + '_' + f2 + '_delta'
        self.feature1 = f1
        self.feature2 = f2
    
    def fit_transform(self, df, **args):
        df[self.name] = df[self.feature1] - df[self.feature2]
    
class div_feature(new_feature):
    ''' Creates feature = feature1 / feature2'''
    feature1 = ''
    feature2 = ''
    
    def __init__(self,f1,f2):
        self.name = f1 + '_' + f2 + '_div'
        self.feature1 = f1
        self.feature2 = f2
    
    def fit_transform(self, df, **args):
        df[self.name] = df[self.feature1] / df[self.feature2]
        
class mult_feature(new_feature):
    ''' Creates feature = feature1 * feature2'''
    feature1 = ''
    feature2 = ''
    
    def __init__(self,f1,f2):
        self.name = f1 + '_' + f2 + '_mult'
        self.feature1 = f1
        self.feature2 = f2
    
    def fit_transform(self, df, **args):
        df[self.name] = df[self.feature1] * df[self.feature2]
        
        

class cluster_feature(new_feature):
    '''Creates a given set of clusters for the given set of features'''
    model = ''
    features = []
    n_clusters = 1
    
    def __init__(self, name):
        self.name = name
        
    def fit_transform(self,df, n_clusters, target_feature, feature_set='all'):
        '''Fits K-Means and transforms the train set, feature_set: list-like or \'all\' '''
        if feature_set == 'all':
            self.features = df.columns.difference([target_feature])
        else:
            self.features = feature_set
        
        self.n_clusters = n_clusters
        
        self.model = sk.cluster.KMeans(n_clusters=self.n_clusters, n_jobs=-1)
        self.model.fit(df[self.features])
        df[self.name] = self.model.predict(df[self.features])
        
    def transform(self,df):
        '''Transforms the test set'''
        df[self.name] = self.model.predict(df[self.features])
