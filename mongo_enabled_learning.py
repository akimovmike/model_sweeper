
import operator
import marshal, types

import pymongo
import json

import pandas as pd
import numpy as np


sources = []
client = pymongo.MongoClient("localhost", 27017)
db = client.dislearn
db.authenticate("[login]", "[password]", source='admin')


db.create_collection('learns')
db.create_collection('learn_tasks')
db.create_collection('models')

db.create_collection('samples')
db.create_collection('index_sets')

db.create_collection('functions')
db.create_collection('model_data')

def test_1_1_feature_generation_function(input_df, params=None):
    
    return input_df.median()

test_1_function_feature_gen_inserted_id = db['functions'].insert_one({
        'Type':'Feature Generation',
        'Code': marshal.dumps(test_1_1_feature_generation_function.__code__),
        'Serialization':'marshall',
        'Libraries':['pd','np']
    }).inserted_id

def learn_bessmislennaya_model(sample_df, train_index, target_var, parameters):
    
    # здесь будет город садъ
    model_bessmislennay_chtoto = lambda x: 1 if x>0.5 else 0
    return model_bessmislennay_chtoto

bessmisslennaya_model_learn_inserted_id = db['functions'].insert_one({
        'Type': 'Learn',
        'Code': marshal.dumps(learn_bessmislennaya_model.__code__),
        'Serialization':'marshall',
        'Libraries':['pd','np', 'xgb']
    }).inserted_id

model_test_1_id = db['models'].insert_one({
        "Model Name": "test_1",
        "Model Description": "test model 1 for functionality test",
        "Initial Sample Location": ('/home/jupyter/jupyter_webroot/kgl/santander/data/train.csv', 'Text File'),
        "Feature Generation Function": (test_1_function_feature_gen_inserted_id, 'functions'),
        "Feature Generation Index": 'ID',
        'Target Variable': 'TARGET',
#         'Feature Generation Params': None,
#         "Feature Evaluation Function": None,
        "Feature Sample Location": ('features_test_1', 'Text File'),
        "Learn Function": (bessmisslennaya_model_learn_inserted_id, 'functions'),
#         "Learn Function Parameters":None,
    }).inserted_id

model_test_1_index_set_train_id = db['index_sets'].insert_one({
        "Index Set":[1,3,5,7,11,13]
    }).inserted_id
model_test_1_index_set_test_id = db['index_sets'].insert_one({
        "Index Set":[2,4,6,8,10,12,]
    }).inserted_id
model_test_1_index_set_validation_id = db['index_sets'].insert_one({
        "Index Set":[9]
    }).inserted_id

model_test_1_learn_1_id=db['learns'].insert_one({
        'Model':(model_test_1_id, 'models'),
        'Feature Generation Params': {'depth':17, 'pni_depth':19, 'feature_set_for_meditation':['A','B']},
        'Train Index': (model_test_1_index_set_train_id, 'index_sets'),
        'Test Index': (model_test_1_index_set_train_id, 'index_sets'),
        'Model':None, 
        'Prediction':None,
        'Evaluation Results':{'AUC':None},
        'Confusion Matrix': None
        }).inserted_id


def sw_compute_features(learn_id):
    
    learn_data = db['learns'].find_one(learn_id)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    
    sample_df = pd.read_csv(model_data['Initial Sample Location'],sep="\t")
    
    feature_generating_function_code = marshal.loads(model_data['Learn Function'])
    
    feature_generating_function = types.FunctionType(learn_function_code, globals(), "onagaeshmas")
    
    return feature_generating_function(sample_df)

def sw_learn(learn_id):
    print(learn_id)
    learn_data = db['learns'].find_one(learn_id)
    print(learn_data)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    
    sample_df = pd.read_csv(model_data['Feature Sample Location'],sep="\t")
    
    learn_function_code = marshal.loads(model_data['Learn Function'])
    
    learn_function = types.FunctionType(learn_function_code, globals(), "onagaeshmas")
    
    return learn_function(sample_df,learn_data['Train Index'], model_data['Target Variable'])