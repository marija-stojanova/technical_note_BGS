from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from scipy.stats import iqr
from sklearn.model_selection import GridSearchCV
import pandas as pd
from functions.error_and_plots import calculate_errors
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, LassoLarsIC
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
import pandas as pd
import matplotlib.pyplot as plt

from functions.error_and_plots import *


def model_grid_search(X_train:pd.DataFrame, y_train:pd.DataFrame, model, k:int = 5, degree:int = 1, sfs:bool=False, n_features:int=11, direction:str = 'forward', model_kargs={}):
    '''perform a k-fold cross validation for an sklearn model fitting the X_train and y_train data with a given degree of polynomial input features and a list 
    of arguments for that model
    the model type (and its arguments) are specified when the function is called
    '''
    if sfs:
        pipe = Pipeline(
            steps=[
                    # ("imputer", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                    ("scaler", StandardScaler()), 
                    ("feature_selector", SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)),
                    ("Model", model),
                ]
        )
    else:
        pipe = Pipeline(
            steps=[
                    # ("imputer", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                    ("scaler", StandardScaler()), 
                    ("Model", model),
                ]
        )
    grid = GridSearchCV(pipe, param_grid=model_kargs, cv=k, scoring='r2')
    grid.fit(X_train, y_train)

    return grid
    
    
def model_logo_grid_search(X, y, model, groups, degree=1, k=5, verbose = False, names=[],  model_kargs={}):
    '''similar to the "model_grid_search" function, with the addition of a Leav-one-group-out procedure. 
    returns only the predicted values on output for the moment (y_predict for the entire response vector)'''
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    y_predict_logo = pd.Series(index=X.index, dtype='float')
    for train_index, test_index in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        grid = model_grid_search(X_train=X_train, y_train=y_train, model=model, degree=degree, sfs=False, k=k, model_kargs=model_kargs)
        y_predict = grid.predict(X_test)
        y_predict_logo.iloc[test_index] = y_predict
        calculate_errors(grid, X_test, y_test, name=names[X_test.index[0]])
    return y_predict_logo

def learn_corrective_model(data:pd.DataFrame, 
                           test_index:list,
                           train_index:list,
                           x_name:list, 
                           y_name:str, 
                           cv:int=5, 
                           model_name:str='LR', 
                           model:BaseEstimator=LinearRegression(),
                           model_parameters:dict={}, 
                           verbose=False,):
    '''Learns a scikit-learn compliant model (default is linear regression) using data[x_name] as features and data[y_name] as target'''

    from pathlib import Path
    Path("output/corrections/" + model_name).mkdir(parents=True, exist_ok=True)

    X = data[data.columns.intersection(x_name)]
    y = data[y_name]
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    regressor = model_grid_search(X_train=X_train, 
                             y_train=y_train, 
                             model=model, 
                             k=cv, 
                             degree=1, 
                             sfs=False, 
                             n_features=7,
                             model_kargs=model_parameters)
    
    if verbose:
        print('Train score = ', r2_score(y_true=y_train, y_pred=regressor.predict(X_train)))
        print('Test score = ', r2_score(y_true=y_test, y_pred=regressor.predict(X_test)))

    return regressor


def run_and_plot_corrective_model(regressor_tic:BaseEstimator, 
                                  regressor_toc:BaseEstimator,
                                  all_istep:pd.DataFrame,
                                  all_vt:pd.DataFrame,
                                  all_re7:pd.DataFrame,
                                  re6_origin:str='istep', 
                                  chn_origin:str='las', 
                                  model_name:str='LR',
                                  features:str='all',
                                  report:list=[],):
    '''Plots the correction of existing corrective models of TIC and TOC'''

    plt.subplots(3, 2, figsize=(10,18))
    # LAS ISTEP
    plt.subplot(3,2,1)
    y_correction_las_istep = regressor_tic.predict(all_istep[regressor_tic.best_estimator_.feature_names_in_])
    y_chn_las_istep = all_istep['TIC'+'_las']
    y_re_las_istep = all_istep['TIC'+ '_' + re6_origin]
        
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_las_istep, y=y_re_las_istep, correction=y_correction_las_istep.ravel(), xlabel='TIC LAS (CHN)', ylabel='TIC ISTeP (RE6)', name='TIC')
    plt.grid()
    report.append([model_name, 'TIC', chn_origin, re6_origin, features, 'las', 'istep', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])

    plt.subplot(3,2,2)
    y_correction_las_istep = regressor_toc.predict(all_istep[regressor_toc.best_estimator_.feature_names_in_])
    y_chn_las_istep = all_istep['TOC'+'_las']
    y_re_las_istep = all_istep['TOC'+ '_' + re6_origin]
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_las_istep, y=y_re_las_istep, correction=y_correction_las_istep.ravel(), xlabel='TOC LAS (CHN)', ylabel='TOC iSTeP (RE6)', name='TOC')
    plt.grid()
    report.append([model_name, 'TOC', chn_origin, re6_origin, features, 'las', 'istep', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])

    # LAS vt
    plt.subplot(3,2,3)
    y_correction_las_vt = regressor_tic.predict(all_vt[regressor_tic.best_estimator_.feature_names_in_])
    y_chn_las_vt = all_vt['TIC'+'_las']
    y_re_las_vt = all_vt['TIC'+ '_' + re6_origin]
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_las_vt, y=y_re_las_vt, correction=y_correction_las_vt.ravel(), xlabel='TIC_las', ylabel='TIC_vt', name='TIC')
    plt.grid()
    report.append([model_name, 'TIC', chn_origin, re6_origin, features, 'las', 'vt', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])


    plt.subplot(3,2,4)
    y_correction_las_vt = regressor_toc.predict(all_vt[regressor_toc.best_estimator_.feature_names_in_])
    y_chn_las_vt = all_vt['TOC'+'_las']
    y_re_las_vt = all_vt['TOC'+ '_' + re6_origin]
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_las_vt, y=y_re_las_vt, correction=y_correction_las_vt.ravel(), xlabel='TOC_las', ylabel='TOC_vt', name='TOC')
    plt.grid()
    report.append([model_name, 'TOC', chn_origin, re6_origin, features,'las', 'vt', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])

    # RE7
    plt.subplot(3,2,5)
    y_correction_re7_las_istep = regressor_tic.predict(all_re7[regressor_tic.best_estimator_.feature_names_in_])
    y_chn_re7_las_istep = all_re7['TIC'+'_las']
    y_re_re7_las_istep = all_re7['TIC'+ '_' + re6_origin]
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_re7_las_istep, y=y_re_re7_las_istep, correction=y_correction_re7_las_istep.ravel(), xlabel='TIC LAS (CHN)', ylabel='TIC VT (RE7)', name='TIC RE7')
    plt.grid()
    report.append([model_name, 'TIC', chn_origin, re6_origin, features,'LAS_RE7', 'vt_RE7', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])


    plt.subplot(3,2,6)
    y_correction_re7_las_istep = regressor_toc.predict(all_re7[regressor_toc.best_estimator_.feature_names_in_])
    y_chn_re7_las_istep = all_re7['TOC'+'_las']
    y_re_re7_las_istep = all_re7['TOC'+ '_' + re6_origin]
    r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected = scatterplot_correction(x=y_chn_re7_las_istep.astype(float), y=y_re_re7_las_istep.astype(float), correction=y_correction_re7_las_istep.ravel(), xlabel='TOC LAS (CHN)', ylabel='TOC VT (RE7)', name='TOC RE7')
    plt.grid()
    report.append([model_name, 'TOC', chn_origin, re6_origin, features,'LAS_RE7', 'vt_RE7', r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected])


    plt.tight_layout()
    return report


def run_full_model(data: pd.DataFrame,
                   re6_predictors:list,
                   re6_origin:str='istep', 
                   chn_origin:str='las', 
                   model_name:str='SVM', 
                   features:str='all', 
                   features_tic:list=[],
                   features_toc:list=[],
                   report:list=[], 
                   only_noncarbonated:bool=False, 
                   only_carbonated:bool=False,
                   TIC_carbonated_threshold:int=2, 
                   ):
       if model_name in ['SVM']:
              model_type = SVR()
              param_grid = {  
                            'Model__kernel': ['rbf'], 
                            'Model__degree': [1],
                            # 'Model__gamma' : [0.001, 0.01, 0.01, 0.1, 1, 10, 100],
                            # 'Model__C' : [0.001, 0.01, 0.01, 0.1, 1, 10, 100]}
                            'Model__gamma' : [0.1, 0.000001],
                            'Model__C' : [0.1, 0.000001, 1]}
       elif model_name == 'RF':
              model_type = RandomForestRegressor()
              param_grid = {  
                     'Model__n_estimators': [100, 500, 1000], 
                     'Model__max_features': [0.33], 
                     'Model__max_depth' : [2, 3, 5],
                     'Model__min_samples_split' : [2, 3, 5, 10],
                     'Model__min_samples_leaf' : [2, 3, 5, 10],
                     }
       elif model_name == 'Ridge':
              model_type = Ridge()
              param_grid = {
                     'Model__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]
              }
       elif model_name in ['LR', 'Linear regression']:
              model_type = LinearRegression()
              param_grid = {}
       
       elif model_name == 'LassoLarsIC':
              model_type = LassoLarsIC()
              param_grid = {}
       else:
              param_grid = {}

      
       data_type = ''
       if only_carbonated:
              data_type = '_carbonated'
       elif only_noncarbonated:
              data_type = '_noncarbonated'

       if features == 'all':
              x_name_tic = x_name_toc = re6_predictors
       elif features == 'selected':
              x_name_tic = ['TIC_'+re6_origin]
              x_name_toc = ['TOC_'+re6_origin]
       else:
              x_name_tic = features_tic
              x_name_toc = features_toc

       X_train, X_test, _, _ = train_test_split(data, data['delta_tic_' + chn_origin + '_' + re6_origin], test_size=0.3, random_state=10)
       train_index = X_train.index
       test_index = X_test.index

       # learn the two corrective models (one for TOC and one for TIC)
       regressor_tic = learn_corrective_model(data=data, 
                                              train_index= train_index,
                                              test_index=test_index,
                                              x_name= x_name_tic,
                                              y_name='delta_tic_' + chn_origin + '_' + re6_origin, 
                                              model_name=model_name, 
                                              model=model_type, 
                                              model_parameters=param_grid)
       regressor_toc = learn_corrective_model(data=data, 
                                              train_index=train_index,
                                              test_index=test_index,
                                              x_name=x_name_toc, 
                                              y_name='delta_toc_' + chn_origin + '_' + re6_origin, 
                                              model_name=model_name, 
                                              model=model_type, 
                                              model_parameters=param_grid)

       all_vt = data.loc[:,(~data.columns.str.contains('_istep', case=False)&(~data.columns.str.contains('_re7', case=False)))]
       all_istep = data.loc[:,(~data.columns.str.contains('_re7', case=False)&(~data.columns.str.contains('_vt', case=False)))]
       all_re7 = data.loc[:,(~data.columns.str.contains('_istep', case=False)&(~data.columns.str.contains('_vt', case=False)))]
       
       if re6_origin == 'vt':
              all_istep.columns = all_istep.columns.str.replace('_istep', '_vt')
              all_re7.columns = all_re7.columns.str.replace('_re7', '_vt')
              all_vt = all_vt.iloc[test_index]
       if re6_origin == 'istep':
              all_vt.columns = all_vt.columns.str.replace('_vt', '_istep')
              all_re7.columns = all_re7.columns.str.replace('_re7', '_istep')
              all_istep = all_istep.iloc[test_index]
       if re6_origin == 're7':
              all_istep.columns = all_istep.columns.str.replace('_istep', '_re7')
              all_vt.columns = all_vt.columns.str.replace('_vt', '_re7')
              all_re7 = all_re7.iloc[test_index]
       
       report = run_and_plot_corrective_model(regressor_tic=regressor_tic, 
                                   regressor_toc=regressor_toc, 
                                   all_vt = all_vt,
                                   all_istep = all_istep,
                                   all_re7 = all_re7,
                                   re6_origin=re6_origin, 
                                   chn_origin=chn_origin,
                                   model_name=model_name,
                                   report=report,
                                   features=features)


       if only_carbonated:
              model_name = model_name + "_only carbonated"
       elif only_noncarbonated:
              model_name = model_name + "_only_noncarbonated"

       Path("output/corrections/"+model_name).mkdir(parents=True, exist_ok=True)
       plt.savefig('output/corrections/'+model_name+'/'+chn_origin + '_' + re6_origin + '_features=' + str(features)+ data_type + '.png', bbox_inches='tight')

       return report, regressor_toc, regressor_tic, train_index, test_index