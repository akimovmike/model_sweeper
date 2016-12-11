import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# a function which will help us create XGBoost models and perform cross-validation
# alg: xgb algorithm initialized by "=XGBClassifier(...)"
# predictors: X
# target: y
# etc...
def modelfit(alg, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(predictors.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,\
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(predictors, target, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(predictors)
    dtrain_predprob = alg.predict_proba(predictors)[:, 1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

# examples of usage
# 1
predictors = X.columns

xgb1 = XGBClassifier(learning_rate=0.1,\
n_estimators=1000,\
max_depth=5,\
min_child_weight=1,\
gamma=0,\
subsample=0.8,\
colsample_bytree=0.8,\
objective='binary:logistic',\
nthread=4,\
scale_pos_weight=1,\
seed=27)

modelfit(xgb1, X, y)

# 2
param_test1=dict(max_depth=[i for i in range(3,10,2)], min_child_weight=[i for i in range(1,6,2)])

gsearch1=GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,\
min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,\
objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),\
param_grid=param_test1, scoring='roc_auc',n_jobs=4, iid=False, cv=5)

gsearch1.fit(X, y)
print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# 3
param_test2=dict(max_depth=[4, 5, 6], min_child_weight=[4, 5, 6])

gsearch2=GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,\
min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,\
objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27), \
param_grid=param_test2, scoring='roc_auc',n_jobs=4, iid=False, cv=5)

gsearch2.fit(X, y)
print(gsearch2.grid_scores_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)

# 4
param_test3=dict(gamma=[i/10.0 for i in range(0, 5)])

gsearch3=GridSearchCV(estimator=XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4,\
min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,\
objective='binary:logistic', nthread=4, scale_pos_weight=1,seed=27),\
param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

gsearch3.fit(X, y)
print(gsearch3.grid_scores_)
print(gsearch3.best_params_)
print(gsearch3.best_score_)