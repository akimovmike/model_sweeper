## pseudocode concept
## (c) Mikhail Akimov, 2016

import sklearn as sk
import hyperopt, joblib, scipy

class model_instance(object):
    name = 'model'
    trained = False
    evaluated = False
    model = NULL
    can_parallel = True
    n_folds = 2
    n_jobs = -1
    param_dict = {}
    
    def __init__(self, model, n_folds=2, n_jobs = -1):
        self.model = # call get class model['model_class']
        self.param_dict = model['param_dict']
        self.can_parallel = model['can_parallel']
        self.n_folds = n_folds
        self.n_jobs = n_jobs
    
    def train(self, X_folds, y_folds):        
        if self.can_parallel:
            for fold in n_folds:
                # call hyperopt parameter sweep (self.param_dict, self.model[fold], X_folds, y_folds, n_jobs=self.n_jobs)
        else:
            # call parallel(hyperopt parameter sweep self.model[fold].train, X_folds, y_folds, self.param_dict self.n_jobs)
        
    def predict(self, X):
        # call predict = blend self.model[].predict(X)
        return predict
    
    def predict_proba(self, X):
        # call predict = blend self.model[].predict_proba(X)
        return predict

def train_models(X, y, n_folds):
    system_cores = # call system get cores number - 1
    
    model_zoo = {
    'logreg':{
        model_class = 'logreg'; # todo: correct class name
        can_parallel = True;
        param_dict = {'C':[.01,10]}
        };
    'rf':{
        model_class = 'rf'; # todo: correct class name
        can_parallel = True;
        param_dict = {'max_tree_depth':[5,100]
        # todo: other hyperparams
        };
    # todo: other models        
    }
    
    trained_models = []
    
    # call parallel n_jobs = system_cores/n_folds 
    #    for model in model_zoo:
    #        model_instance = model_instance(model_zoo_parallel[model])
    #        model_instance.train(X, y)
    #        trained_models.append(model_instance)

    return trained_models
    
def predict(trained_models, X):
    preds = {}    
    for model in trained_models:
        preds[model.name] = model.predict(X)
    return preds
    
def predict_proba(trained_models, X):
    preds = {}    
    for model in trained_models:
        preds[model.name] = model.predict_proba(X)
    return preds
    
def blend_preds(preds):
    # preds_df = convert preds to DF
    # check dispersion of preds
    # drop similar preds
    # preds_blend = preds_df.gmean(axis=1)
    return preds_blend
    
def blend_preds_weighted(preds, X, y):
    # check dispersion of preds
    # drop similar preds
    # train blender model/scipy.optimize(X,y)
    # weighted_avg = predict by blender model from filtered preds
    return weighted_avg
        