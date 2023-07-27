from sklearn.metrics import roc_auc_score #AUC
from sklearn.metrics import fbeta_score #F2-score
from scipy.stats import ks_2samp #KS test
from hmeasure import h_score #H-measure

from optbinning import BinningProcess
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np #Array handling
import math #Rounding
import scipy.stats #Skewness
import matplotlib.pyplot as plt #Plotting
import pandas as pd #Data frame
import pickle #Saving models

class Evaluate():
    '''
    Class evaluating the performance of model given predictions and true values
    '''
    def __init__(self, true_vals, preds, def_perc_train):
        self.preds = preds #Store the predictions
        self.true_vals = true_vals #Store the true values
        self.def_perc_train = def_perc_train #Store the percentage of defaults in the training set
        self.results = self.evaluate(true_vals, preds, def_perc_train)

    def f2_score(self, true_vals, preds, def_perc_train):
        '''
        Function calculating the F2 score. Threshold is set such that the number of classified observations match the training set
        '''
        thres = -np.sort(-preds)[math.ceil(def_perc_train * preds.shape[0])] #Find the threshold which results in the same number of classified examples as in the training set
        return fbeta_score(true_vals, (preds >= thres).astype(int), beta = 2) #Return the F2-score
        
    def brier_score(self, true_vals, preds):
        '''
        Function calculating the Brier score based on estimated probabilities and true values
        '''
        return (1 / preds.shape[0]) * np.sum((preds - true_vals) ** 2)

    def ks_test(self, true_vals, preds):
        '''
        Function calculating the value of the Kolmogorov-Smirnov statistic
        '''
        return ks_2samp(preds[true_vals == 0], preds[true_vals == 1]).statistic

    def partial_gini_index(self, true_vals, preds, b = 0.4):
        '''
        Function calculating the partial gini index
        '''
        try:
            return 2 * roc_auc_score(true_vals[preds <= b], preds[preds <= b]) - 1
        except ValueError: #In case only one class is available for true values
            return None
    
    def evaluate(self, true_vals, preds, def_prec_train):
        '''
        Function applying all defined evaluation metrics
        '''
        res_dict = {} #Define a dictionary for the results
        res_dict['AUC'] = roc_auc_score(true_vals, preds) #Calcuate AUC
        res_dict['F2-score'] = self.f2_score(true_vals, preds, def_prec_train) #Calcuate the F2-score
        res_dict['Brier score'] = self.brier_score(true_vals, preds) #Calculate the Brier score
        res_dict['KS statistic'] = self.ks_test(true_vals, preds) #Calculate the KS statistic
        res_dict['Partial GINI Index'] = self.partial_gini_index(true_vals, preds) #Calculate the Partial GINI Index
        try:
            res_dict['H-measure'] = h_score(true_vals, preds) #Calculate the H-measure
        except TypeError: #In case true values are not a numpy array
            res_dict['H-measure'] = h_score(true_vals.values, preds) #Calculate the H-measure

        return res_dict
    
    def permutation_test(self, preds1, preds2 = None, true_vals = None, n_samples = 5000, seed = 69, plot = None):
        '''
        Function performing the permutation test
        '''
        #Check specified predictions
        if (preds2 is None) & (true_vals is None): #If second predictions and true values are not supplied, use the one contained within the class
            preds2 = self.preds
            true_vals = self.true_vals
        elif (preds2 is not None) & (true_vals is not None): #If they are supplied continue normally
            pass
        else: #If only one is supplied, raise an error
            raise Exception('Specify second predictions and true values or neither')
        
        #Check dimensions
        if preds1.shape[0] != preds2.shape[0] != true_vals.shape[0]:
            raise Exception('The number of observations is inconsistent')
        
        #Bootstrap predictions
        generator = np.random.RandomState(seed) #Initiate a generator
        ids = generator.choice(2, (n_samples, preds1.shape[0]), replace = True) #Generate ids for all repetitions
        preds = np.stack([preds1, preds2], axis = 1)[range(preds1.shape[0]), ids] #Put the predictions into a single array
        
        #Perform evaluation
        res = np.apply_along_axis(lambda x: self.evaluate(true_vals, x, self.def_perc_train), 1, preds) #Apply the function to each column
        res = pd.DataFrame(list(res)) #Create a Data Frame from the results
        res1 = self.evaluate(true_vals, preds1, self.def_perc_train) #Get the performance of the model in question

        #Export plots if required
        if plot:
            for metric in res.columns: #Loop through the metrics
                if res1[metric] is not None:
                    plt.hist(res[metric], 'auto') #Plot a histogram
                    plt.title(f'{metric} for {n_samples} repetitions') #Add a title to the plot
                    plt.xlabel(metric) #Add a label for the x axis
                    plt.axvline(res1[metric], color = 'red', label = 'Binned model') #Add a vertical line with the result of the model in question
                    plt.legend(loc = 'upper center') #Add a legend
                    plt.savefig(f'{plot}_{metric}.png') #Save the plot
                    plt.close() #Close the current plot

        #Perform the test
        out = {} #Empty dict for the results
        for metric in res.columns:
            if res1[metric] is None: #In case the partial gini index cannot be evaluated
                out[metric] = None
            if metric == 'Brier score': #The smaller Brier score the better => different alternative hypothesis
                out[metric] = np.mean(res[metric] <= res1[metric]) #Get a percentage of cases where the mixed predictions are better
            else: #For the remaining metrics the larger the better
                out[metric] = np.mean(res[metric] >= res1[metric]) #Get a percentage of cases where the mixed predictions are better
        
        return out

class OneHotBinning(BaseEstimator, TransformerMixin):
    '''
    Wrapper for the optimal binning process and subsequent one hot encoding of the bins
    '''
    def fit(self, X, y=None):
        '''
        Fit the tranformer
        '''
        self.vars = X.columns.to_list()
        self.binning = BinningProcess(self.vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in self.vars}, max_pvalue = 0.1)
        self.one_hot = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
        self.one_hot.fit(self.binning.fit_transform(X, y, metric = 'bins'))
        return self

    def transform(self, X, y=None):
        '''
        Transform the given array
        '''
        X_transf = self.binning.transform(X, metric = 'bins')
        return self.one_hot.transform(X_transf)

def LogReg(df, df_name, kind, C = np.logspace(-4, 4, 50), model_name = 'LogReg', missings = False, one_hot = False):
    '''
    Function wrapper for Logistic Regression with binning and grid search
    '''
    #Specify variables
    indep_vars = df.columns.to_list()
    indep_vars.remove('Target') #Drop the dependent variable form the list
    num_vars = list(np.array(indep_vars)[df.loc[:, indep_vars].nunique() > 2]) #Get numerical variables
    num_vars_ind = [df.columns.to_list().index(i) for i in num_vars] #Convert strings to indices
    cat_vars_ind = list(set(list(range(df.shape[1]))) - set(num_vars_ind)) #Get indices of categorical variables
    cat_vars_ind.remove(df.columns.to_list().index('Target')) #Drop the index of the target

    #Fit the binned model
    if missings: #For estimation with missings
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind), ('imputation', SimpleImputer(strategy = 'most_frequent'), cat_vars_ind)], remainder = 'passthrough') ), ('model', LogisticRegression(max_iter = 1000, random_state = 42))])
    elif one_hot: #For one hot encoding estimation (without missings)
        model_binned_pipe = Pipeline([('binning', BinningProcess(indep_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in indep_vars}, max_pvalue = 0.1)), ('encoding', OneHotEncoder(drop = 'first', handle_unknown = 'ignore')), ('model', LogisticRegression(max_iter = 1000, random_state = 42))])
    else: #For main estimation
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind)], remainder = 'passthrough') ), ('model', LogisticRegression(max_iter = 1000, random_state = 42))])
    params_binned = [{'model__C' : C, 'model__penalty' : ['l2'], 'model__n_jobs' : [-1]}, {'model__C' : C, 'model__penalty' : ['l1'], 'model__solver' : ['liblinear']}, {'model__penalty' : [None], 'model__n_jobs' : [-1]}]
    model_binned = GridSearchCV(model_binned_pipe, params_binned, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_binned.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with pd.ExcelWriter(f'Binning summary/{model_name}_{df_name}_{kind}.xlsx') as writer: #Save binning summary
        for i in num_vars: #Loop through the numerical variables
            if one_hot:
                model_binned.best_estimator_['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
            else:
                model_binned.best_estimator_['binning'].transformers_[0][1]['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
    pd.DataFrame(model_binned.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_binned_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_binned_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_binned.best_estimator_, handle)
    
    #Fit the raw model
    if missings: #In case missings need to be imputed
        cat_cols = list(np.where(df.loc[:, indep_vars].nunique() <= 2)[0]) #Get indices of categorical columns
        mean_cols = [] #Empty list for columns to be impute by mean
        median_cols = [] #Empty list for columns to be impute by median
        for i in set(list(range(len(indep_vars)))) - set(cat_cols): #Loop through all numerical variables
            if abs(scipy.stats.skew(df.loc[:, indep_vars].iloc[:, i], nan_policy = 'omit')) > 2: #For skewed distributions use median
                median_cols.append(i)
            else: #For symetric use mean
                mean_cols.append(i)
        col_transf = ColumnTransformer([(metric, SimpleImputer(strategy = metric), inds) for metric, inds in zip(['mean', 'median', 'most_frequent'], [mean_cols, median_cols, cat_cols]) if inds != []])
        model_raw_pipe = Pipeline([('imputation', col_transf), ('model', LogisticRegression(max_iter = 1000))])
        params_raw = [{'model__C' : C, 'model__penalty' : ['l2'], 'model__n_jobs' : [-1]}, {'model__C' : C, 'model__penalty' : ['l1'], 'model__solver' : ['liblinear']}, {'model__penalty' : [None], 'model__n_jobs' : [-1]}]
    else:
        model_raw_pipe = LogisticRegression(max_iter = 1000, random_state = 42)
        params_raw = [{'C' : C, 'penalty' : ['l2'], 'n_jobs' : [-1]}, {'C' : C, 'penalty' : ['l1'], 'solver' : ['liblinear']}, {'penalty' : [None], 'n_jobs' : [-1]}]
    model_raw = GridSearchCV(model_raw_pipe, params_raw, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_raw.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    pd.DataFrame(model_raw.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_raw_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_raw_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_raw.best_estimator_, handle)

    return model_binned.best_estimator_, model_raw.best_estimator_

def DecTree(df, df_name, kind, crit = ['gini', 'entropy'], depth = [5, 10, 20, 50], max_leaves = [25, 50, 100, 500], model_name = 'DecTree', missings = False, one_hot = False):
    '''
    Function wrapper for decision tree including binning and grid search
    '''
    #Specify variables
    indep_vars = df.columns.to_list()
    indep_vars.remove('Target') #Drop the dependent variable form the list
    num_vars = list(np.array(indep_vars)[df.loc[:, indep_vars].nunique() > 2]) #Get numerical variables
    num_vars_ind = [df.columns.to_list().index(i) for i in num_vars] #Convert strings to indices
    cat_vars_ind = list(set(list(range(df.shape[1]))) - set(num_vars_ind)) #Get indices of categorical variables
    cat_vars_ind.remove(df.columns.to_list().index('Target')) #Drop the index of the target

    #Fit the binned model
    if missings:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind), ('imputation', SimpleImputer(strategy = 'most_frequent'), cat_vars_ind)], remainder = 'passthrough') ), ('model', DecisionTreeClassifier(random_state = 42))])
    elif one_hot:
        model_binned_pipe = Pipeline([('binning', BinningProcess(indep_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in indep_vars}, max_pvalue = 0.1)), ('encoding', OneHotEncoder(drop = 'first', handle_unknown = 'ignore')), ('model', DecisionTreeClassifier(random_state = 42))])
    else:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind)], remainder = 'passthrough') ), ('model', DecisionTreeClassifier(random_state = 42))])
    params_binned = [{'model__criterion' : crit, 'model__max_depth' : depth}, {'model__criterion' : crit, 'model__max_leaf_nodes' : max_leaves}]
    model_binned = GridSearchCV(model_binned_pipe, params_binned, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_binned.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with pd.ExcelWriter(f'Binning summary/{model_name}_{df_name}_{kind}.xlsx') as writer: #Save binning summary
        for i in num_vars: #Loop through the numerical variables
            if one_hot:
                model_binned.best_estimator_['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
            else:
                model_binned.best_estimator_['binning'].transformers_[0][1]['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
    pd.DataFrame(model_binned.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_binned_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_binned_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_binned.best_estimator_, handle)
    
    #Fit the raw model
    if missings: #In case missings need to be imputed
        cat_cols = list(np.where(df.loc[:, indep_vars].nunique() <= 2)[0]) #Get indices of categorical columns
        mean_cols = [] #Empty list for columns to be impute by mean
        median_cols = [] #Empty list for columns to be impute by median
        for i in set(list(range(len(indep_vars)))) - set(cat_cols): #Loop through all numerical variables
            if abs(scipy.stats.skew(df.loc[:, indep_vars].iloc[:, i], nan_policy = 'omit')) > 2: #For skewed distributions use median
                median_cols.append(i)
            else: #For symetric use mean
                mean_cols.append(i)
        col_transf = ColumnTransformer([(metric, SimpleImputer(strategy = metric), inds) for metric, inds in zip(['mean', 'median', 'most_frequent'], [mean_cols, median_cols, cat_cols]) if inds != []])
        model_raw_pipe = Pipeline([('imputation', col_transf), ('model', DecisionTreeClassifier(random_state = 42))])
        params_raw = [{'model__criterion' : crit, 'model__max_depth' : depth}, {'model__criterion' : crit, 'model__max_leaf_nodes' : max_leaves}]
    else:
        model_raw_pipe = DecisionTreeClassifier(random_state = 42)
        params_raw = [{'criterion' : crit, 'max_depth' : depth}, {'criterion' : crit, 'max_leaf_nodes' : max_leaves}]
    model_raw = GridSearchCV(model_raw_pipe, params_raw, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_raw.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    pd.DataFrame(model_raw.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_raw_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_raw_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_raw.best_estimator_, handle)

    return model_binned.best_estimator_, model_raw.best_estimator_

def GaussNB(df, df_name, kind, model_name = 'GaussNB', missings = False, one_hot = False):
    '''
    Function wrapper for the Gaussian Naive Bayes including binning and grid search
    '''
    #Specify variables
    indep_vars = df.columns.to_list()
    indep_vars.remove('Target') #Drop the dependent variable form the list
    num_vars = list(np.array(indep_vars)[df.loc[:, indep_vars].nunique() > 2]) #Get numerical variables
    num_vars_ind = [df.columns.to_list().index(i) for i in num_vars] #Convert strings to indices
    cat_vars_ind = list(set(list(range(df.shape[1]))) - set(num_vars_ind)) #Get indices of categorical variables
    cat_vars_ind.remove(df.columns.to_list().index('Target')) #Drop the index of the target

    #Fit the binned model
    if missings:
        model_binned = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind), ('imputation', SimpleImputer(strategy = 'most_frequent'), cat_vars_ind)], remainder = 'passthrough') ), ('model', GaussianNB())])
    elif one_hot:
        model_binned = Pipeline([('binning', BinningProcess(indep_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in indep_vars}, max_pvalue = 0.1)), ('encoding', OneHotEncoder(drop = 'first', handle_unknown = 'ignore')), ('model', BernoulliNB())])
    else:
        model_binned = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind)], remainder = 'passthrough') ), ('model', GaussianNB())])
    model_binned.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with pd.ExcelWriter(f'Binning summary/{model_name}_{df_name}_{kind}.xlsx') as writer: #Save binning summary
        for i in num_vars: #Loop through the numerical variables
            if one_hot:
                model_binned['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
            else:
                model_binned['binning'].transformers_[0][1]['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
    with open(f'Models\{model_name}_{df_name}_binned_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_binned, handle)
    
    #Fit the raw model
    if missings: #In case missings need to be imputed
        cat_cols = list(np.where(df.loc[:, indep_vars].nunique() <= 2)[0]) #Get indices of categorical columns
        mean_cols = [] #Empty list for columns to be impute by mean
        median_cols = [] #Empty list for columns to be impute by median
        for i in set(list(range(len(indep_vars)))) - set(cat_cols): #Loop through all numerical variables
            if abs(scipy.stats.skew(df.loc[:, indep_vars].iloc[:, i], nan_policy = 'omit')) > 2: #For skewed distributions use median
                median_cols.append(i)
            else: #For symetric use mean
                mean_cols.append(i)
        col_transf = ColumnTransformer([(metric, SimpleImputer(strategy = metric), inds) for metric, inds in zip(['mean', 'median', 'most_frequent'], [mean_cols, median_cols, cat_cols]) if inds != []])
        model_raw = Pipeline([('imputation', col_transf), ('model', GaussianNB())])
    else:
        model_raw = GaussianNB()
    model_raw.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with open(f'Models\{model_name}_{df_name}_raw_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_raw, handle)

    return model_binned, model_raw

def NN(df, df_name, kind, alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10], activation = ['relu', 'logistic', 'tanh'], model_name = 'NN', missings = False, one_hot = False):
    '''
    Function wrapper for a one-layer Neural Network
    '''
    #Specify variables
    indep_vars = df.columns.to_list()
    indep_vars.remove('Target') #Drop the dependent variable form the list
    num_vars = list(np.array(indep_vars)[df.loc[:, indep_vars].nunique() > 2]) #Get numerical variables
    num_vars_ind = [df.columns.to_list().index(i) for i in num_vars] #Convert strings to indices
    cat_vars_ind = list(set(list(range(df.shape[1]))) - set(num_vars_ind)) #Get indices of categorical variables
    cat_vars_ind.remove(df.columns.to_list().index('Target')) #Drop the index of the target

    #Fit the binned model
    if missings:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind), ('imputation', SimpleImputer(strategy = 'most_frequent'), cat_vars_ind)], remainder = 'passthrough') ), ('model', MLPClassifier(max_iter = 500, random_state = 42))])
    elif one_hot:
        model_binned_pipe = Pipeline([('binning', BinningProcess(indep_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in indep_vars}, max_pvalue = 0.1)), ('encoding', OneHotEncoder(drop = 'first', handle_unknown = 'ignore')), ('model', MLPClassifier(max_iter = 500, random_state = 42))])
    else:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind)], remainder = 'passthrough') ), ('model', MLPClassifier(max_iter = 500, random_state = 42))])
    params_binned = {'model__alpha' : alphas, 'model__hidden_layer_sizes' : [[math.ceil(2 * len(indep_vars) / 3)], [math.ceil(len(indep_vars) / 2)], [len(indep_vars)]], 'model__activation' : activation}
    model_binned = GridSearchCV(model_binned_pipe, params_binned, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_binned.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with pd.ExcelWriter(f'Binning summary/{model_name}_{df_name}_{kind}.xlsx') as writer: #Save binning summary
        for i in num_vars: #Loop through the numerical variables
            if one_hot:
                model_binned.best_estimator_['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
            else:
                model_binned.best_estimator_['binning'].transformers_[0][1]['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
    pd.DataFrame(model_binned.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_binned_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_binned_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_binned.best_estimator_, handle)
    
    #Fit the raw model
    if missings: #In case missings need to be imputed
        cat_cols = list(np.where(df.loc[:, indep_vars].nunique() <= 2)[0]) #Get indices of categorical columns
        mean_cols = [] #Empty list for columns to be impute by mean
        median_cols = [] #Empty list for columns to be impute by median
        for i in set(list(range(len(indep_vars)))) - set(cat_cols): #Loop through all numerical variables
            if abs(scipy.stats.skew(df.loc[:, indep_vars].iloc[:, i], nan_policy = 'omit')) > 2: #For skewed distributions use median
                median_cols.append(i)
            else: #For symetric use mean
                mean_cols.append(i)
        col_transf = ColumnTransformer([(metric, SimpleImputer(strategy = metric), inds) for metric, inds in zip(['mean', 'median', 'most_frequent'], [mean_cols, median_cols, cat_cols]) if inds != []])
        model_raw_pipe = Pipeline([('imputation', col_transf), ('model', MLPClassifier(max_iter = 500, random_state = 42))])
        params_raw = {'model__alpha' : alphas, 'model__hidden_layer_sizes' : [[math.ceil(2 * len(indep_vars) / 3)], [math.ceil(len(indep_vars) / 2)], [len(indep_vars)]], 'model__activation' : activation}
    else:
        model_raw_pipe = MLPClassifier(max_iter = 500, random_state = 42)
        params_raw = {'alpha' : alphas, 'hidden_layer_sizes' : [[math.ceil(2 * len(indep_vars) / 3)], [math.ceil(len(indep_vars) / 2)], [len(indep_vars)]], 'activation' : activation}
    model_raw = GridSearchCV(model_raw_pipe, params_raw, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_raw.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    pd.DataFrame(model_raw.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_raw_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_raw_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_raw.best_estimator_, handle)

    return model_binned.best_estimator_, model_raw.best_estimator_

def RandForest(df, df_name, kind, criterion = ['gini', 'entropy'], n_estimators = [10, 50, 100, 500], max_leaves = [25, 50, 100, 500], depth = [5, 10, 20, 50], model_name = 'RandForest', missings = False, one_hot = False):
    '''
    Function wrapper for Random Forest including binning and grid search
    '''
    #Specify variables
    indep_vars = df.columns.to_list()
    indep_vars.remove('Target') #Drop the dependent variable form the list
    num_vars = list(np.array(indep_vars)[df.loc[:, indep_vars].nunique() > 2]) #Get numerical variables
    num_vars_ind = [df.columns.to_list().index(i) for i in num_vars] #Convert strings to indices
    cat_vars_ind = list(set(list(range(df.shape[1]))) - set(num_vars_ind)) #Get indices of categorical variables
    cat_vars_ind.remove(df.columns.to_list().index('Target')) #Drop the index of the target

    #Fit the binned model
    if missings:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind), ('imputation', SimpleImputer(strategy = 'most_frequent'), cat_vars_ind)], remainder = 'passthrough') ), ('model', RandomForestClassifier(random_state = 42, n_jobs = -1))])
    elif one_hot:
        model_binned_pipe = Pipeline([('binning', BinningProcess(indep_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in indep_vars}, max_pvalue = 0.1)), ('encoding', OneHotEncoder(drop = 'first', handle_unknown = 'ignore')), ('model', RandomForestClassifier(random_state = 42, n_jobs = -1))])
    else:
        model_binned_pipe = Pipeline([('binning', ColumnTransformer([('binning', Pipeline([('binning', BinningProcess(num_vars, n_jobs = -1, binning_fit_params = {i:{'monotonic_trend': 'auto_asc_desc'} for i in num_vars}, max_pvalue = 0.1))]), num_vars_ind)], remainder = 'passthrough') ), ('model', RandomForestClassifier(random_state = 42, n_jobs = -1))])
    params_binned = [{'model__criterion' : criterion, 'model__n_estimators' : n_estimators, 'model__max_depth' : depth}, {'model__criterion' : criterion, 'model__n_estimators' : n_estimators, 'model__max_leaf_nodes' : max_leaves}]
    model_binned = GridSearchCV(model_binned_pipe, params_binned, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_binned.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    with pd.ExcelWriter(f'Binning summary/{model_name}_{df_name}_{kind}.xlsx') as writer: #Save binning summary
        for i in num_vars: #Loop through the numerical variables
            if one_hot:
                model_binned.best_estimator_['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
            else:
                model_binned.best_estimator_['binning'].transformers_[0][1]['binning'].get_binned_variable(i).binning_table.build().to_excel(writer, sheet_name = i[:min(31, len(i))])
    pd.DataFrame(model_binned.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_binned_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_binned_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_binned.best_estimator_, handle)
    
    #Fit the raw model
    if missings: #In case missings need to be imputed
        cat_cols = list(np.where(df.loc[:, indep_vars].nunique() <= 2)[0]) #Get indices of categorical columns
        mean_cols = [] #Empty list for columns to be impute by mean
        median_cols = [] #Empty list for columns to be impute by median
        for i in set(list(range(len(indep_vars)))) - set(cat_cols): #Loop through all numerical variables
            if abs(scipy.stats.skew(df.loc[:, indep_vars].iloc[:, i], nan_policy = 'omit')) > 2: #For skewed distributions use median
                median_cols.append(i)
            else: #For symetric use mean
                mean_cols.append(i)
        col_transf = ColumnTransformer([(metric, SimpleImputer(strategy = metric), inds) for metric, inds in zip(['mean', 'median', 'most_frequent'], [mean_cols, median_cols, cat_cols]) if inds != []])
        model_raw_pipe = Pipeline([('imputation', col_transf), ('model', RandomForestClassifier(random_state = 42, n_jobs = -1))])
        params_raw = [{'model__criterion' : criterion, 'model__n_estimators' : n_estimators, 'model__max_depth' : depth}, {'model__criterion' : criterion, 'model__n_estimators' : n_estimators, 'model__max_leaf_nodes' : max_leaves}]
    else:
        model_raw_pipe = RandomForestClassifier(random_state = 42, n_jobs = -1)
        params_raw = [{'criterion' : criterion, 'n_estimators' : n_estimators, 'max_depth' : depth}, {'criterion' : criterion, 'n_estimators' : n_estimators, 'max_leaf_nodes' : max_leaves}]
    model_raw = GridSearchCV(model_raw_pipe, params_raw, scoring = 'roc_auc', n_jobs = -1, cv = 3)
    model_raw.fit(df.loc[:, indep_vars], df.loc[:, 'Target'])

    #Store data
    pd.DataFrame(model_raw.cv_results_).to_excel(f'Models/Grid search/{model_name}_{df_name}_raw_{kind}.xlsx', index = False) #Save grid search results
    with open(f'Models\{model_name}_{df_name}_raw_{kind}.pkl', 'wb') as handle: #Store the model
        pickle.dump(model_raw.best_estimator_, handle)

    return model_binned.best_estimator_, model_raw.best_estimator_

def predict_binned(model, X):
    '''
    A function generating predictions based on an estimated model and input X
    '''
    num_cols = model['binning'].transformers_[0][2] #Get indices of numerical columns
    X_transf = model['binning'].transformers_[0][1]['binning'].transform(X.iloc[:, num_cols], metric = 'woe', metric_missing = 'empirical') #Transform numerical columns
    cat_cols = list(set(list(range(X.shape[1]))) - set(num_cols)) #Get categorical features
    if cat_cols != []: #If categorical variables are present, attach them
        X_cat_transf = model['binning'].transformers_[1][1].transform(X.iloc[:, cat_cols]) #Impute categorical features
        X_transf = np.hstack((X_transf, X_cat_transf)) #Append categorical vars
    return model['model'].predict_proba(X_transf)[:, 1] #Generate predictions