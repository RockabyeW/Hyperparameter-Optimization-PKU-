from hyperopt import fmin, tpe, hp

from sklearn import datasets

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt



def hyperopt_fun(X, y,params):
    '''
    Hyperopt的目标函数，调参优化的依据
    params:params,Hyperopt.hp,参数空间，调参将根据参数空间进行优化
    '''
    params=argsDict_tranform(params)
    alg=RandomForestRegressor(**params)
    metric = cross_val_score(
        alg,
        X,
        y,
        cv=10,scoring="neg_mean_squared_error")
    return min(-metric)
    def hyperopt_space():
        import hyperopt.pyll
        from hyperopt.pyll import scope
        space= {
                'n_estimators':hp.randint("n_estimators_RF", 300),
                'max_depth':hp.randint("max_depth_RF", 35),
                'min_samples_split':hp.uniform('min_samples_split_RF',0.4,0.7),
                'min_samples_leaf':hp.randint('min_samples_leaf_RF',300),
                'min_weight_fraction_leaf':hp.uniform('min_weight_fraction_leaf_RF',0,0.5),
                'max_features':hp.uniform('max_features_RF',0.5,1.0),
                'oob_score':True,
                'n_jobs':-1,
                'random_state':2019
            }
        return space
    def argsDict_tranform(argsDict,isPrint=False,best=False):
        if best:
            ### 对获取到的最后调优结果进行转换参数

            argsDict['n_estimators']=argsDict.pop('n_estimators'+'_%s'%t)+1
            argsDict['max_depth']=argsDict.pop('max_depth'+'_%s'%t)+7
            argsDict['min_samples_leaf']=argsDict.pop('min_samples_leaf'+'_%s'%t)+100
            argsDict['min_samples_split']=argsDict.pop('min_samples_split'+'_%s'%t)
            argsDict['min_weight_fraction_leaf']=argsDict.pop('min_weight_fraction_leaf'+'_%s'%t)
            argsDict['max_features']=argsDict.pop('max_features'+'_%s'%t)
        else:
            ###调参过程中，对于采样空间的处理，例如有些参数不能为0之类的情况
            argsDict['n_estimators']=argsDict['n_estimators']+1
            argsDict['max_depth']=argsDict['max_depth']+7
            argsDict['min_samples_leaf']=argsDict['min_samples_leaf']+100
        if isPrint:
            print(argsDict)
        else:
            pass
        return argsDict
    def hyperopt_hpo(max_evals=100):
        algo=partial(tpe.suggest,n_startup_jobs=1)
        space=hyperopt_space()
        trials=Trials()
        best=fmin(hyperopt_fun,space,algo=algo,trials=trials,max_evals=max_evals, pass_expr_memo_ctrl=None)
        print(best)
        best_t=argsDict_tranform(best,best=True)
        return best_t