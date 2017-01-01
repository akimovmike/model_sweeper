
import operator
import marshal, types

import pymongo
import json

import numpy as np
import pandas as pd


sources = []
client = pymongo.MongoClient("host", 'port')
db = client.dislearn
db.authenticate("login", "password", source='login')


import os

import ml_metrics

import sklearn as sk

from sklearn import metrics
from bson import ObjectId

def check_sample_exists(sample_notation):
    if sample_notation[-1]=='Text File':
        return os.path.exists(sample_notation[0])

def load_df_from_sample_notation(sample_notation, **kvargs):
    if sample_notation[-1]=='Text File':
        if type(sample_notation[1])==dict:
            kvargs.update(sample_notation[1])
            
        if not 'sep' in kvargs:
            kv_args['sep'] = "\t"
        return pd.read_csv(sample_notation[0] , **kvargs)
    
def save_df_to_sample_notation(input_df, sample_notation):
    if sample_notation[-1]=='Text File':
        input_df.to_csv(sample_notation[0] ,sep="\t", index=False)

def sw_compute_features(learn_data, overwrite_existing=False, worker_id=None):
    
#     learn_data = db['learns'].find_one(learn_id)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    
    
#     sample_df = load_df_from_sample_notation(model_data['Initial Sample Location'])
    
    if not check_sample_exists(model_data['Feature Sample Location']) or overwrite_existing:

        feature_generating_function_code = marshal.loads(db[model_data['Feature Generation Function'][-1]]\
                                                  .find_one(model_data['Feature Generation Function'][0])['Code'])
            

        feature_generating_function = types.FunctionType(feature_generating_function_code, globals())
        
        # save_df_to_sample_notation(, model_data['Feature Sample Location'])
        learn_data = feature_generating_function(learn_data, model_data)
        
    learn_data['Status']['Features Computed'] = True
    db['learns'].update(learn_data['_id'], learn_data)

def sw_learn(learn_data, overwrite_existing, worker_id=None):
#     learn_data = db['learns'].find_one(learn_id)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    if learn_data['Status']['Features Computed']:
        if not 'Fitted Model' in learn_data or learn_data['Fitted Model']==None or overwrite_existing:
#             sample_df = load_df_from_sample_notation(model_data['Feature Sample Location'])
            learn_function_code = marshal.loads(db[model_data['Learn Function'][-1]]\
                                                  .find_one(model_data['Learn Function'][0])['Code'])
            learn_function = types.FunctionType(learn_function_code, globals())
            learn['Fitted Model'] = learn_function(learn_data, model_data)

        learn_data['Status']['Model Fitted'] = True
        
        db['learns'].update(learn_data['_id'], learn_data)

def sw_compute_prediction(learn_data, overwrite_existing, worker_id=None):
#     learn_data = db['learns'].find_one(learn_id)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    
    if learn_data['Status']['Model Fitted']:
        if not 'Prediction' in learn_data or learn_data['Prediction']==None or overwrite_existing:
#             sample_df = load_df_from_sample_notation(model_data['Feature Sample Location'])
            predict_function_code = marshal.loads(db[model_data['Prediction Computation Function'][-1]]\
                                                  .find_one(model_data['Prediction Computation Function'][0])['Code'])
            predict_function = types.FunctionType(prediction_compute_function_code, globals())
            learn['Prediction'] = predict_function(learn_data, model_data)

        learn_data['Status']['Prediction Computed'] = True
        db['learns'].update(learn_data['_id'], learn_data)

def compute_metrics(metrics, learn_data, model_data):
    target_label_gound_truth = load_df_from_sample_notation(model_data['Feature Sample Location'])[model_data['Target Variable']]
    prediction = learn_data['Prediction']
    if metric=='AUC':
        return 0.9 # zaglushka sk.metrics.auc_mathafaka( target_label_gound_truth, prediction)
    
def sw_evalute_model(learn_data, overwrite_existing, worker_id=None):
#     learn_data = db['learns'].find_one(learn_id)
    model_data = db[learn_data['Model'][-1]].find_one(learn_data['Model'][0])
    if learn_data['Status']['Prediction Computed']:
        for metric in learn_data['Evaluation Results'].keys():
            if learn_data['Evaluation Results']==None or overwrite_existing:
                learn_data['Evaluation Results'][metrics] = compute_metrics(metric, learn_data, model_data)    

        learn_data['Status']['Model Evaluated'] = True
        db['learns'].update(learn_data['_id'], learn_data)

stage_to_func_dict={
    'Features Computed':sw_compute_features,
    'Model Fitted':sw_learn,
    'Prediction Computed':sw_compute_prediction,
    'Model Evaluated': sw_evalute_model
}

                      
def report_on_set_of_learns(set_of_learns):
    return pd.concat([pd.DataFrame(learn['Status'], index=[learn['_id']]) for learn in learns], axis=0).mean(),\
                      pd.concat( [pd.DataFrame(learn['Evaluation Results'], index=[learn['_id']]) \
                                  for learn in learns], axis=0).mean()
                      
                      
def sw_report(task_id):
    
    learns_set = list( db['learns'].find_many(
            {'Parent Task _id':(task_id, 'learn_tasks')}
        ) )
    
    model_learn_dict = {}
    for learn in learns_set:
        if learn['Model'] in model_learn_dict.keys():
            model_learn_dict[learn['Model']].append(learn)
        else:
            model_learn_dict[learn['Model']] = [learn]
            
    for model, learns_set in model_learn_dict.iteritems():
        print(str(model), report_on_set_of_learns(learns_set)) 
        

def zaglushka_compute_features(learn_data, model_data):
    initial_sample_df = load_df_from_sample_notation(model_data['Initial Sample Location'])

    
    save_df_to_sample_notation(initial_sample_df.group_by(model_data["Feature Generation Index"]).apply(lambda x:x.iloc[0])\
                               .set_index(model_data["Feature Generation Index"], drop=False), 
                               model_data['Feature Sample Location'])
    
    return learn_data
    
def zaglushka_learn(learn_data, model_data):
    sample_df = load_df_from_sample_notation(model_data['Feature Sample Location'])
    

    
    save_df_to_sample_notation(sample_df.group_by(model_data["Feature Generation Index"]).apply(lambda x:x.iloc[0]), 
                               model_data['Feature Sample Location'])
    
    return learn_data
    
def zaglushka_predict(learn_data, model_data):
    sample_df = load_df_from_sample_notation(model_data['Feature Sample Location'])
    
    
    return list(np.random.random(sample_df.shape[0]))
        

def sw_create_cv_task(task_data, model_data, learn_data, **kvargs):
    
    # сериализация функций и сохранение баз
    for data in [task_data, model_data, learn_data]:
        for k, object_to_serialize in data.items():
            if type(object_to_serialize) == type(check_sample_exists): #just fucntion, no matter which one
                lookup_in_db = db['functions'].find_one({'Name':object_to_serialize.__name__})
                if lookup_in_db:
                    data[k] = (lookup_in_db['_id'],'functions')
                else:
                    data[k] = (db['functions'].insert_one({
                                    'Name': object_to_serialize.__name__,
                                    'Type': k,
                                    'Code': marshal.dumps(object_to_serialize.__code__),
                                    'Serialization':'marshall',
                                    'Libraries':['pd','np']
                                }).inserted_id,
                               'functions')
    
            if type(object_to_serialize) == pd.core.frame.DataFrame:\
                ## make filename from model name
                new_filename = model_data['Name'].replace(" ","_")+"_df_"+k.replace(" ","_")+".txt"
                ## test there is no file there, no file
                
                ## save file
                
                ## save notation to smple database if insert to database
                
                ## change 
                save_df_to_sample_notation(object_to_serialize, (new_filename, 'Text File') )
                data[k] = (new_filename, 'Text File')

    ## form CV_Set if not set
    if 'CV Set' not in task_data:
        n_folds = kvargs['n_folds'] if 'n_folds' in kvargs else 3
        cv_type = kvargs['cv_type'] if 'cv_type' in kvargs else 'train_test'
        
        unique_indexes = load_df_from_sample_notation(model_data['Initial Sample Location'])\
            [model_data["Feature Generation Index"]].unique()
        
        if cv_type == 'train_test':
            
            
            task_data['CV Set'] = [{'Train Index':db['index_sets'].insert_one({'Index Set':[str(unique_indexes[i]) for i in x[0]]}).inserted_id,
                                    'Test Index':db['index_sets'].insert_one({'Index Set':[str(unique_indexes[i]) for i in x[1]]}).inserted_id} for x in KFold( n_folds).split(unique_indexes)]
    
    print(task_data, model_data, learn_data)
    learn_data['Model'] = (db['models'].insert_one(model_data).inserted_id,'models')
    learn_data['Parent Task _id'] = (db['learn_tasks'].insert_one(task_data).inserted_id, 'learn_tasks')
    task_data['_id'] = learn_data['Parent Task _id']
    
    assert 'CV Set' in task_data
    
    # go through CV set and save learns, adding Parent 
    for cv_split_data in task_data['CV Set']:
        learn_data['CV Split']=cv_split_data
        if '_id' in learn_data:
            learn_data.pop('_id')
        learn_id = db['learns'].insert_one(learn_data).inserted_id
    
    return task_data['_id']


def sw_worker(task_id, worker_id):
    
    
    task_data = db['learn_tasks'].find_one( (task_id[0] if type(task_id)==tuple else task_id) )
    print(task_data)
    learns_set = list( db['learns'].find({'Parent Task _id':task_id}))
    
    for learn in learns_set:
        if 'Lock' not in db['learns'].find_one(learn['_id']) or db['learns'].find_one(learn['_id'])['Lock']==worker_id:
            db['learns'].update_one({'_id':learn['_id']},{'$set':{'Lock':worker_id}})
            
            for stage in task_data['Stages']:
                if not learn['Status'][stage]:
                      stage_to_func_dict[stage](learn, task_data['Overwrite Existed'], worker_id)
                        
            db['learns'].update_one({'_id':learn['_id']},{'$delete':'Lock'})
                      

# sw_worker(
#     sw_create_cv_task(task_data={
#         'Stages':['Features Computed','Model Fitted','Prediction Computed','Model Evaluated'],
#         'Overwrite Existed':False
#     },model_data={
#         "Model Name": "test_0",
#         "Model Description": "test model 0 for functionality test",
#         "Initial Sample Location": ('/home/jupyter/jupyter_webroot/kgl/santander/data/train.csv',{'sep':','}, 'Text File'),
#         "Feature Generation Function": zaglushka_compute_features,
#         "Feature Generation Index": 'ID',
#         'Target Variable': 'TARGET',
# #         'Feature Generation Params': None,
# #         "Feature Evaluation Function": None,
#         "Feature Sample Location": ('features_test_0.txt', 'Text File'),
#         "Learn Function": zaglushka_learn,
# #         "Learn Function Parameters":None,
#         "Predict Function": zaglushka_predict, 
#     },learn_data={
#         'Status':{'Features Computed':False,'Model Fitted':False,'Prediction Computed':False,'Model Evaluated':False}
#     }),
    
#     1)