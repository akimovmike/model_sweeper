import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import sklearn.metrics
import scipy.stats as sc

def calibrate(res):
    '''Calibrates probabilities'''
    f1=np.log(1+res)
    f2=np.sin(res)
    f3=f1*f1
    f4=f2*f2
    f5=f1*f2
    return (res+f1+f2+f3+f4+f5)/4.025

def make_ensemble(series, name, use_gmean = True):
    '''series: list of pandas.Series
       name: string'''
    preds = pd.concat(series, axis=1)
    if use_gmean:
        ens = preds.apply(lambda x: sc.gmean(x), axis=1)
    else:
        ens = preds.apply(lambda x: np.mean(x), axis=1) # an alternative    
    ens = pd.Series(ens,index = series[0].index, name = 'probability')
    ens.to_csv(name+'_ens.csv', index_label = 't_id', header = True)
    
def test_multicollinearity(df, target_name, r2_threshold = 0.89):
    '''Tests if any of the features could be predicted from others with R2 >= 0.89
    
    input: dataframe, name of target (to exclude)
    
   '''
    r2s = pd.DataFrame()
    for feature in df.columns.difference([target_name]):
        model = sk.linear_model.Ridge()
        model.fit(df[df.columns.difference([target_name,feature])], df[feature])
        
        pos = np.in1d(model.coef_, np.sort(model.coef_)[-5:])

        r2s = r2s.append(pd.DataFrame({'r2':sk.metrics.r2_score(df[feature],\
			model.predict(df[df.columns.difference([target_name, feature])])),\
			'predictors' : str(df.columns.difference([target_name, feature])[np.ravel(np.argwhere(pos == True))].tolist())}, index = [feature]))
        print('Testing', feature)

    print('-----------------')

    if len(r2s[r2s['r2'] >= r2_threshold]) > 0:
        print('Multicollinearity detected')
        print(r2s[r2s['r2'] >= r2_threshold])
    else:
        print('No multicollinearity')