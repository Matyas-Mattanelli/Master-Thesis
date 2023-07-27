import pandas as pd
import os

means = [] #Empty list to store the target means
indices = [] #Empty list to store the indices

for i in os.listdir('D:/My stuff/School/Master/Master Thesis/Data/Train data'): #Loop through all training files
    df = pd.read_csv(f'D:/My stuff/School/Master/Master Thesis/Data/Train data/{i}') #Load the data set
    means.append(df['Target'].mean()) #Calcuate the mean of the dependent variable and store it
    indices.append(i.removesuffix('.csv')) #Derive an index for the mean

pd.Series(means, index = indices, name = 'Mean').to_excel('Train_means.xlsx', index_label = 'Data set') #Export to excel
