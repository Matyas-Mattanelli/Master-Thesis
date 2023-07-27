import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

#Specify parameters
seed = 42
test_size = 0.2

#Create the partitions
x = [f'{i}_no_outs.csv' for i in ['GiveMeSomeCredit', 'HomeCredit', 'CreditApproval', 'CreditCardTaiwan', 'SouthGermanCredit']]
for df_name in os.listdir('D:/My stuff/School/Master/Master Thesis/Data/Final data'): #Loop through all data sets
    df = pd.read_csv(f'D:/My stuff/School/Master/Master Thesis/Data/Final data/{df_name}') #Load the data set
    df_train, df_test = train_test_split(df, test_size = test_size, random_state = seed) #Train-test split
    mask = df_train.nunique() > 1 #Create a mask for columns to retain
    print(f'{sum(np.invert(mask))} constant variables disregarded from {df_name}')
    df_train = df_train.loc[:, mask] #Disregard constant variables
    df_test = df_test.loc[:, mask] #Disregard constant variables
    df_train.to_csv(f'D:/My stuff/School/Master/Master Thesis/Data/Train data/{df_name}', index = False) #Store the train data
    df_test.to_csv(f'D:/My stuff/School/Master/Master Thesis/Data/Test data/{df_name}', index = False) #Store the test data