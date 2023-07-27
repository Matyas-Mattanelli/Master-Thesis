import pandas as pd
import numpy as np
import pickle
import os
import datetime

import tools

#Specify the parameters
folder = None
kind = 'one_hot'
estimation = True
evaluation = True
shutdown = True
C = np.logspace(-4, 4, 50)
crit = ['gini', 'entropy']
depth = [5, 10, 20, 50]
max_leaves = [25, 50, 100, 500]
n_estimators = [10, 50, 100, 500]
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
activation = ['relu', 'logistic', 'tanh']

#Define the output folder
if folder is not None:
    if not os.path.isdir(folder): #If a path is provided, check its validity
        folder = None
if folder is None: #If no folder is provided or the folder does not exist, create a new folder
    folder = f'Run {datetime.datetime.now():%Y%m%d%H%M} {kind}' #Specify the folder
    os.makedirs(folder) #Create the folder
    for i in ['Binning summary', 'Evaluation', 'Models']: os.makedirs(f'{folder}/{i}') #Create subfolders
    os.makedirs(f'{folder}/Evaluation/Plots') #Create subfolders
    os.makedirs(f'{folder}/Models/Grid search') #Create subfolders

#Change the working directory
os.chdir(folder)

#Define model functions
model_names = {'LogReg' : lambda df, df_name: tools.LogReg(df, df_name, kind, C, one_hot = True),
                'DecTree' : lambda df, df_name : tools.DecTree(df, df_name, kind, crit, depth, max_leaves, one_hot = True), 
                'RandForest' : lambda df, df_name : tools.RandForest(df, df_name, kind, crit, n_estimators, max_leaves, depth, one_hot = True), 
                'GaussNB' : lambda df, df_name : tools.GaussNB(df, df_name, kind, one_hot = True), 
                'NN' : lambda df, df_name : tools.NN(df, df_name, kind, alphas, activation, one_hot = True)}

#Load the data sets
df_names = ['HomeCredit', 'CreditApproval', 'CreditCardTaiwan', 'SouthGermanCredit', 'GiveMeSomeCredit']
df_paths_train = [f'D:/My stuff/School/Master/Master Thesis/Data/Train data/{i}_no_cats.csv' for i in df_names[:-1]] + [f'D:/My stuff/School/Master/Master Thesis/Data/Train data/GiveMeSomeCredit_main.csv']
df_paths_test = [f'D:/My stuff/School/Master/Master Thesis/Data/Test data/{i}_no_cats.csv' for i in df_names[:-1]] + [f'D:/My stuff/School/Master/Master Thesis/Data/Test data/GiveMeSomeCredit_main.csv']

if evaluation:
    #Load means of the dependent variable in the training sample
    train_means = pd.read_excel('D:\My stuff\School\Master\Master Thesis\Project\Train_means.xlsx', index_col = 'Data set')

    #Prepare output holders
    evals = [] #List for the dictionaries of results
    indices = [] #List of indices containing the model/dataset/data combinations
    perm_tests = [] #List for the dictionaries of permutation test results
    indices_perm = [] #List of indices containing the model/dataset combinations for the permutation test

for idx, df_path_test in enumerate(df_paths_test): #Loop through the data sets
    
    if evaluation:
        #Load the data set
        df_test = pd.read_csv(df_path_test)

        #Specify variables
        indep_vars = df_test.columns.to_list()
        indep_vars.remove('Target') #Drop the dependent variable form the list

    for model_name in model_names: #Loop through the models
        if estimation: #Estimate the models if required
            df_train = pd.read_csv(df_paths_train[idx]) #Load the training data set
            model_binned, model_raw = model_names[model_name](df_train, df_names[idx])
            print(f'Estimation of {model_name} for {df_names[idx]} done')
        elif evaluation: #Otherwise load already estimated models
            #Load the models
            with open(f'Models/{model_name}_{df_names[idx]}_binned_{kind}.pkl', 'rb') as handle: #Binned model
                model_binned = pickle.load(handle)
            with open(f'Models/{model_name}_{df_names[idx]}_raw_{kind}.pkl', 'rb') as handle: #Raw model
                model_raw = pickle.load(handle)
            print(f'{model_name} for {df_names[idx]} loaded')

        if evaluation:
            #Get predictions
            preds_binned = model_binned.predict_proba(df_test.loc[:, indep_vars])[:, 1]
            preds_raw = model_raw.predict_proba(df_test.loc[:, indep_vars])[:, 1]

            #Get evaluation metrics
            eval_binned = tools.Evaluate(df_test['Target'], preds_binned, train_means.loc[df_paths_train[idx].split('/')[-1].replace('.csv', '')].values[0]) #Binned data
            eval_raw = tools.Evaluate(df_test['Target'], preds_raw, train_means.loc[df_paths_train[idx].split('/')[-1].replace('.csv', '')].values[0]) #Raw data

            #Append the results
            evals.extend([eval_binned.results, eval_raw.results])
            indices.extend([f'{model_name}_{df_names[idx]}_{i}' for i in ['binned', 'raw']])

            #Perform the permutation test
            perm_test = eval_raw.permutation_test(preds_binned, plot = f'Evaluation/Plots/{model_name}_{df_names[idx]}_{kind}')
            perm_tests.append(perm_test)
            indices_perm.append(f'{model_name}_{df_names[idx]}')

            #Print info
            print(f'Evaluation of {model_name} for {df_names[idx]} done')

if evaluation:
    #Create a dataframe from the results
    evals_df = pd.DataFrame(evals, index = indices)
    perm_tests_df = pd.DataFrame(perm_tests, index = indices_perm)
    with pd.ExcelWriter(f'Evaluation/{kind}.xlsx') as writer: #Initiate an excel workbook
        evals_df.to_excel(writer, sheet_name = 'Metrics') #Add a sheet with the metrics
        perm_tests_df.to_excel(writer, sheet_name = 'Permutation test') #Add a sheet with the permutation test

    #Get parameters of optimal models from the grid search
    file_names = os.listdir('Models/Grid search') #Get the filenames
    model_names = set([i.split('_')[0] for i in file_names]) #Get model names
    res_dict = {i:[] for i in model_names} #Prepare a dicitonary for the results
    for file_name in file_names: #Get parameters of the best model from each folder
        df = pd.read_excel(f'Models/Grid search/{file_name}') #Load the grid search results
        params = eval(df.loc[df['rank_test_score'] == 1, 'params'].values[0]) #Get params of best model
        new_params = {} #Prepare a new dictionary with adjusted keys
        new_params['Data set'] = file_name.split('_')[1] #Add data set name
        new_params['Model'] = file_name.split('_')[2] #Add model type
        for key in params: #Strip the keys of the model prefix in case of binned model
            if 'model__' in key:
                new_key = key.replace('model__', '')
            else:
                new_key = key
            if new_key not in ['solver', 'n_jobs']: #Skip technical parameters
                new_params[new_key] = params[key]
        new_params['Average AUC'] = round(df.loc[df['rank_test_score'] == 1, 'mean_test_score'].values[0], 3) #Get the average AUC from the 3-folds
        res_dict[file_name.split('_')[0]].append(new_params) #Append the results to the final dictionary for the given model
    with pd.ExcelWriter(f'Best_parameters_{kind}.xlsx') as writer: #Initiate an excel file
        for model in model_names: #Loop through the models
            df = pd.DataFrame(res_dict[model])
            df.columns = [i.title().replace('_', ' ') for i in df.columns] #Adjust the column names
            df.to_excel(writer, sheet_name = model, index = False) #Export to a given sheet
    with open(f'Best_parameters_latex_{kind}.txt', 'w') as handle: #Initiate an excel file
        for model in model_names: #Loop through the models
            df = pd.DataFrame(res_dict[model])
            df.columns = [i.title().replace('_', ' ') for i in df.columns] #Adjust the column names
            handle.write(f'%{model} hyperparameters\n') #Add a new line for clarity
            handle.write(df.to_latex(index = False, na_rep = '', caption = f'{model} hyperparameters', label = f'tab:{model}params', position = '!htbp'))
            handle.write('\n')

    #Export results to latex
    #Raw metrics
    evals_df = evals_df.round(3) #Round the results to three decimals
    #Add columns for data set and model type
    metric_names = evals_df.columns.to_list()
    evals_df['Method'] = [i.split('_')[0] for i in evals_df.index]
    evals_df['Data set'] = [i.split('_')[1] for i in evals_df.index]
    evals_df['Type'] = [i.split('_')[2] for i in evals_df.index]
    evals_df = evals_df.loc[:, ['Method', 'Data set', 'Type'] + metric_names] #Move indicator columns in the beginning
    evals_df.sort_values(['Method', 'Data set', 'Type'], inplace = True)
    #Separate the results by models and export to latex
    with open(f'Results_latex_{kind}.txt', 'w') as handle: #Initiate an excel file
        for model_name in evals_df['Method'].unique(): #Loop through all models
            df = evals_df.loc[evals_df['Method'] == model_name, :].copy() #Subset the original data set
            df.drop('Method', axis = 1, inplace = True) #Drop the obsolete model column
            df.reset_index(drop = True, inplace = True) #Disregard the index
            binned_avg = df.loc[df['Type'] == 'binned', metric_names].mean().to_list() #Calculate average across data sets for binned models
            raw_avg = df.loc[df['Type'] == 'raw', metric_names].mean().to_list() #Calculate average across data sets for raw models
            df.loc[df.shape[0], :] = ['binned > raw', '-'] + (df.loc[df['Type'] == 'binned', metric_names].reset_index(drop = True) > df.loc[df['Type'] == 'raw', metric_names].reset_index(drop = True)).sum().to_list()
            df.loc[df.shape[0] - 1, 'Brier score'] = (df.loc[df['Type'] == 'binned', 'Brier score'].reset_index(drop = True) < df.loc[df['Type'] == 'raw', 'Brier score'].reset_index(drop = True)).sum() #Adjust Brier score since the lower the better
            df.loc[df.shape[0], :] = ['Average', 'binned'] + binned_avg #Append average for binned model
            df.loc[df.shape[0], :] = ['Average', 'raw'] + raw_avg #Append average for raw model
            handle.write(f'%{model_name} results ({kind})\n') #Add a new line for clarity
            handle.write(df.to_latex(index = False, na_rep = '', caption = f'{model_name} results ({kind})', label = f'tab:{model_name}res{kind}', position = '!htbp'))
            handle.write('\n')
    #Permutation test
    perm_tests_df = perm_tests_df.round(3) #Round the results to three decimals
    #Add columns for data set and model type
    metric_names = perm_tests_df.columns.to_list()
    perm_tests_df['Method'] = [i.split('_')[0] for i in perm_tests_df.index]
    perm_tests_df['Data set'] = [i.split('_')[1] for i in perm_tests_df.index]
    perm_tests_df = perm_tests_df.loc[:, ['Method', 'Data set'] + metric_names] #Move indicator columns in the beginning
    perm_tests_df.sort_values(['Method', 'Data set'], inplace = True)
    #Separate the results by models and export to latex
    with open(f'Permutation_tests_latex_{kind}.txt', 'w') as handle: #Initiate an excel file
        for model_name in perm_tests_df['Method'].unique(): #Loop through all models
            df = perm_tests_df.loc[perm_tests_df['Method'] == model_name, :].copy() #Subset the original data set
            df.drop('Method', axis = 1, inplace = True) #Drop the obsolete model column
            df.reset_index(drop = True, inplace = True) #Disregard the index
            df.loc[df.shape[0], :] = ['p <= 0.05'] + (df.loc[:, metric_names] <= 0.05).sum().to_list()
            handle.write(f'%{model_name} permutation tests ({kind})\n') #Add a new line for clarity
            handle.write(df.to_latex(index = False, na_rep = '', caption = f'{model_name} permutation tests ({kind})', label = f'tab:{model_name}perm{kind}', position = '!htbp'))
            handle.write('\n')

#Shutdown the computer if required
if shutdown:
    import time
    import os
    time.sleep(100) #Let all processes finish
    os.system('shutdown -s') #Command to shutdown the computer