import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from scipy.stats import iqr
import pandas as pd


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 13
# define custom colormaps
norm = matplotlib.colors.Normalize(-1,1)

# summer colors divergent
color1d = '#003f5c'
color2d = '#4f697e'
color3d = '#8a96a2'
color4d = '#c6c6c6'
color5d = '#e1a9a3'
color6d = '#f38982'
color7d = '#ff6361'
color8dyellow = '#ffc16a' #an extra shade of yellow
color9dgray = '#2a2b2d'
colors_summer = [[norm(-1.0), color1d],
          [norm(-0.6), color2d],
          [norm(-0.3), color3d],
          [norm( 0.0), color4d],
          [norm( 0.3), color5d],
          [norm( 0.6), color6d],
          [norm( 1.0), color7d]]
cmap_summer_divergent = matplotlib.colors.LinearSegmentedColormap.from_list("", colors_summer)
colorlist_summer_divergent = [color1d, color7d, color3d, color5d, color2d, color6d, color4d, color8dyellow, color9dgray]

sns.set_palette(colorlist_summer_divergent)


################################################
############## ERROR FUNCTIONS #################
################################################

def rmspe_function (y_true, y_pred):
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

def rrmse_function (y_true, y_pred):
    return mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False) / np.mean(y_true) 

def rpd_function(y_true, y_pred):
    '''computes the Ratio of Performance to InterQuartile distance (RPIQ), which is defined as 
    interquartile range of the observed values divided by the Root Mean Square Error or Prediction (RMSEP).
    '''
    return np.std(y_true)/mean_squared_error(y_true, y_pred, squared=False)


def rpiq_function  (y_true, y_pred):
    IQR = iqr(y_true)
    # RMSPE = rmspe_function(y_true=y_true, y_pred=y_pred)
    RMSE = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    return IQR/RMSE


def bias_function (y_true, y_pred):
    return y_pred.mean() - y_true.mean()


def calculate_errors(grid, X_true, y_true, y_pred=[], name='', verbose:bool=False):
    '''calculates and prints the R2 score, MAE,... for a given estimator'''
    
    if not len(y_pred):
        y_pred = grid.predict(X_true)
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    bias = bias_function(y_true, y_pred)
    rpiq = rpiq_function(y_true, y_pred)
    rpd = rpd_function(y_true, y_pred)
    ve = explained_variance_score(y_pred=y_pred, y_true=y_true)

    if (name and verbose):
        print(name)
    if verbose:
        print("R2 score: ", r2)
        print("Mean absolute error: ", mae)
        print("RMSE: ", rmse)
        print("Bias: ", bias)
        print("RPIQ: ", rpiq)
        print("RPD: ", rpd)
        print("VE: ", ve)
        print()
    return r2, mae, rmse, bias, rpiq, rpd, ve


def r2_rmse_per_site(g):
    r2 = r2_score(g['Actual'], g['Predicted'])
    rmse = np.sqrt(mean_squared_error(g['Actual'], g['Predicted']))
    return pd.Series(dict(r2 = r2, rmse = rmse))


####################################################################
######################## PLOT FUNCTIONS ############################
####################################################################


def plot_correlations (predictors, response_series, response_name, method='spearman'):
    # plots the correlation of type "method" for the response variable and all the predictor variables
    df_correlations = pd.concat([predictors, response_series], axis=1)
    correlations = df_correlations.corr(method=method, numeric_only=True)[response_name].sort_values()
    correlations.drop(correlations.tail(1).index,inplace=True)

    plt.figure(figsize=(15,8))
    plt.bar(x=correlations.index, height=correlations)
    plt.grid()
    plt.axhline(0.5, ls='dotted', color=color7d, lw=4)
    plt.axhline(-0.5, ls='dotted', color=color7d, lw=4)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Correlation with cut-off threshold for feature selection, method is " + method, fontsize=20)
    plt.show()

    return correlations
    
    
    
def plot_model(y_test, y_predict, sites, palette='rainbow'):
    # plots the model in a "Truth vs Prediction" scatter plot. Adds the "ideal" prediction line too
    order = sites.unique()[sites.unique().argsort()]
    sns.scatterplot(x=y_test, y=y_predict, alpha=0.8, s=200, linewidth=0.1, hue=sites, palette=palette, style=sites, hue_order=order)
    plt.plot(y_test, y_test, label='Ideal', c='k')
    plt.grid()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10,10)
    plt.xlabel('Ground truth', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)

    # to make the legend markers larger
    lgnd = plt.legend(loc='lower right', fontsize=15)
    for i in range(len(sites.unique())):
        lgnd.legend_handles[i]._sizes = [100]
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 
    
    
    
def triple_plot(y_internal_predicted, y_internal, sites_internal, y_logo_predicted, y_logo, sites_logo, y_new_data_predicted, y_new_data, sites_new_data, model_name, model_params=[]):
    # it takes three real and predicted y vectors: for internal validation, for leave-one-site-out validation, and for prediction on new data
    # then it plots three scatter plots with the corresponding R2s and RMSEs
    text_size = 20

    plt.subplot(1,3,1)
    plot_model(y_predict=y_internal_predicted, y_test=y_internal, sites=sites_internal)
    plt.title(model_name +' : Internal validation', fontsize=text_size)
    plt.text(.1,.81, 'R2 = {:.2f}'.format(r2_score(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.1,.85, 'RMSE = {:.2f}'.format(np.sqrt(mean_squared_error(y_internal, y_internal_predicted))), fontsize=text_size)
    plt.text(.1,.89, 'RPIQ = {:.2f}'.format(rpiq_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.1,.93, 'Bias = {:.2f}'.format(bias_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.1,.97, 'RPD = {:.2f}'.format(rpd_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.xlabel('Ground truth', fontsize=text_size)
    plt.ylabel('Prediction', fontsize=text_size)
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.subplot(1,3,2)
    plt.title(model_name +' : Leave-one-site-out', fontsize=text_size)
    plot_model(y_predict=y_logo_predicted, y_test=y_logo, sites=sites_logo)
    plt.text(.1,.81, 'R2 = {:.2f}'.format(r2_score(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.1,.85, 'RMSE = {:.2f}'.format(np.sqrt(mean_squared_error(y_logo, y_logo_predicted))), fontsize=text_size)
    plt.text(.1,.89, 'RPIQ = {:.2f}'.format(rpiq_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.1,.93, 'Bias = {:.2f}'.format(bias_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.1,.97, 'RPD = {:.2f}'.format(rpd_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.xlabel('Ground truth', fontsize=text_size)
    plt.ylabel('Prediction', fontsize=text_size)
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    plt.subplot(1,3,3)
    plt.title(model_name +' : Predict on new data', fontsize=text_size)
    plot_model(y_predict=y_new_data_predicted, y_test=y_new_data, sites=sites_new_data)
    plt.text(.1,.81, 'R2 = {:.2f}'.format(r2_score(y_new_data, y_new_data_predicted)), fontsize=text_size)
    plt.text(.1,.85, 'RMSE = {:.2f}'.format(np.sqrt(mean_squared_error(y_new_data, y_new_data_predicted))), fontsize=text_size)
    plt.text(.1,.89, 'RPIQ = {:.2f}'.format(rpiq_function(y_new_data, y_new_data_predicted)), fontsize=text_size)
    plt.text(.1,.93, 'Bias = {:.2f}'.format(bias_function(y_new_data, y_new_data_predicted)), fontsize=text_size)
    plt.text(.1,.97, 'RPD = {:.2f}'.format(rpd_function(y_new_data, y_new_data_predicted)), fontsize=text_size)
    plt.xlabel('Ground truth', fontsize=text_size)
    plt.ylabel('Prediction', fontsize=text_size)
    plt.xlim([0,1])
    plt.ylim([0,1])

    if(model_params):
        print(model_params)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(25,10)
    
    
    
def double_plot(y_internal_predicted, y_internal, sites_internal, y_logo_predicted, y_logo, sites_logo, model_name, model_params=[]):
    # it takes three real and predicted y vectors: for internal validation, for leave-one-site-out validation, and for prediction on new data
    # then it plots three scatter plots with the corresponding R2s and RMSEs
    text_size = 20

    plt.subplot(1,2,1)
    plot_model(y_predict=y_internal_predicted, y_test=y_internal, sites=sites_internal)
    plt.title(model_name +' : Internal validation', fontsize=text_size)
    plt.text(.65,.01, 'R2 = {:.2f}'.format(r2_score(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.65,.05, 'RMSE = {:.2f}'.format(np.sqrt(mean_squared_error(y_internal, y_internal_predicted))), fontsize=text_size)
    plt.text(.65,.09, 'RPIQ = {:.2f}'.format(rpiq_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.65,.13, 'Bias = {:.2f}'.format(bias_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.text(.65,.17, 'RPD = {:.2f}'.format(rpd_function(y_internal, y_internal_predicted)), fontsize=text_size)
    plt.xlabel('Ground truth', fontsize=text_size)
    plt.ylabel('Prediction', fontsize=text_size)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(fontsize=13)#, loc='lower right')
    plt.subplot(1,2,2)    
    plt.title(model_name +' : Leave-one-site-out', fontsize=text_size)
    plot_model(y_predict=y_logo_predicted, y_test=y_logo, sites=sites_logo)
    plt.text(.65,.01, 'R2 = {:.2f}'.format(r2_score(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.65,.05, 'RMSE = {:.2f}'.format(np.sqrt(mean_squared_error(y_logo, y_logo_predicted))), fontsize=text_size)
    plt.text(.65,.09, 'RPIQ = {:.2f}'.format(rpiq_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.65,.13, 'Bias = {:.2f}'.format(bias_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.text(.65,.17, 'RPD = {:.2f}'.format(rpd_function(y_logo, y_logo_predicted)), fontsize=text_size)
    plt.xlabel('Ground truth', fontsize=text_size)
    plt.ylabel('Prediction', fontsize=text_size)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(fontsize=13)#, loc='lower left')

    if(model_params):
        print(model_params)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18,10)

def scatterplot_comparison(data:pd.DataFrame, x_name:'str', y_name:'str', hue:'str'=''):
    from scipy import stats
    from sklearn.metrics import r2_score, mean_squared_error

    if len(hue) > 0:
        sns.scatterplot(data=data, x=x_name ,y=y_name, s=[70], alpha=0.9, hue=hue, palette=cmap_summer_divergent)
    else:
        sns.scatterplot(data=data, x=x_name ,y=y_name, s=[70], alpha=0.9, palette=cmap_summer_divergent)

    plt.grid()
    x = data[x_name].astype(float)
    y = data[y_name].astype(float)
    slope, intercept, _, _, _ = stats.linregress(x=x, y=y)
    r2 = r2_score(y_true=x, y_pred=y)
    rmse = mean_squared_error(y_true=x, y_pred=y, squared=False)
    bias = bias_function(y_true=x, y_pred=y)
    text_size = 12

    plt.text(max(x)/1.7, 2.2*max(x)/10, '{:.2f} + {:.2f} * x'.format(intercept, slope), fontsize=text_size)
    plt.text(max(x)/1.7, 1.5*max(x)/10,'Bias = {:.2f}'.format(bias), fontsize=text_size)
    plt.text(max(x)/1.7, 0.8*max(x)/10,'R2 = {:.2f}'.format(r2), fontsize=text_size)
    plt.text(max(x)/1.7, 1, 'RMSE = {:.2f}'.format(rmse), fontsize=text_size)

def scatterplot_correction(x, y, correction, name=[], ylabel=[], xlabel=[]):
    from scipy import stats

    sns.scatterplot(x=x ,y=y, s=[80], label='Original', alpha=0.6, color=color7d)
    sns.scatterplot(x=x ,y=y + correction, s=[80], label='Corrected', alpha=0.8, color=color1d)
    sns.lineplot(x=x, y=x, label='1:1 line')
    text_size = 15
    plt.title(name, fontsize=text_size+2)
    if ylabel:
        plt.ylabel(ylabel + ' [gC/kg]', fontsize=text_size)
    if xlabel:
        plt.xlabel(xlabel + ' [gC/kg]', fontsize=text_size)
    plt.grid()
    plt.legend(loc='upper left', fontsize=text_size)
    slope, intercept, _, _, _ = stats.linregress(x=x, y=y)
    r2 = r2_score(y_true=x, y_pred=y)
    rmse = mean_squared_error(y_true=x, y_pred=y, squared=False)
    bias = bias_function(y_true=x, y_pred=y)
    rpiq = rpiq_function(y_true=x, y_pred=y)
    

    plt.text(max(x)/1.5, max(x)/3 + abs(min(x)), '{:.2f} + {:.2f} * x'.format(intercept, slope), fontsize=text_size)
    plt.text(max(x)/1.5, max(x)/3.5 + abs(min(x)),'Bias = {:.2f}'.format(bias), fontsize=text_size)
    plt.text(max(x)/1.5, max(x)/4.1 + abs(min(x)),'R2 = {:.2f}'.format(r2), fontsize=text_size)
    plt.text(max(x)/1.5, max(x)/4.8 + abs(min(x)),'RMSE = {:.2f}'.format(rmse), fontsize=text_size)

    slope_corrected, intercept_corrected, _, _, _ = stats.linregress(x=x, y=y+correction)
    r2_corrected = r2_score(y_true=x, y_pred=y + correction)
    rmse_corrected = mean_squared_error(y_true=x, y_pred=y + correction, squared=False)
    bias_corrected = bias_function(y_true=x, y_pred=y+correction)
    rpiq_corrected = rpiq_function(y_true=x, y_pred=y+correction)

    plt.text(max(x)/1.5, max(x)/8 + abs(min(x)), '{:.2f} + {:.2f} * x'.format(intercept_corrected, slope_corrected), fontsize=text_size)
    plt.text(max(x)/1.5, max(x)/12 + abs(min(x)),'Bias = {:.2f}'.format(bias_corrected), fontsize=text_size)
    plt.text(max(x)/1.5, max(x)/25 + abs(min(x)),'R2 = {:.2f}'.format(r2_corrected), fontsize=text_size)
    plt.text(max(x)/1.5, 0 + abs(min(x)),'RMSE = {:.2f}'.format(rmse_corrected), fontsize=text_size)

    return r2, bias, rmse, rpiq, slope, intercept, r2_corrected, bias_corrected, rmse_corrected, rpiq_corrected, intercept_corrected, slope_corrected
    