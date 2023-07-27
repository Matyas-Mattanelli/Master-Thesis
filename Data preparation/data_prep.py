import logging
import datetime

class Dataset():
    '''
    Class accepting a dataset and preparing it for estimation. Returns two versions of the data set, one for binning and one for raw usage.
    '''

    def __init__(self, df, name, dep_var, thres_corr, thres_miss, test_size):
        '''
        Constructor calling all functions to transform the dataset and store it
        '''
        #Configure logging
        logging.basicConfig(filename = f'{name} {datetime.datetime.now():%Y%m%d%H%M}.log', encoding = 'utf-8', level = logging.INFO)

        self.raw_df = df #Store the raw data set
        self.dep_var = dep_var #Store the name of the dependent variable
        self.thres_miss = thres_miss #Store the threshold for the maximum percentage of missing values
        self.thres_corr = thres_corr #Store the correlation threshold for feature selection
        self.test_size = test_size #Store the test (and possibly validation) size
    
    def drop_incomplete(self, df, thres):
        '''
        Function removing all fetaures with a percentage of missing values higher than given threshold
        '''
        missing_perc = df.isna().sum() / df.shape[0] #Get missing percentages for all columns
        cols_to_drop = missing_perc[missing_perc >= thres].to_list() #Get names of columns to drop
        if cols_to_drop == []: #If no columns are to be dropped, return the original data set
            logging.info('No columns were dropped due to incompleteness')
            return df
        else:
            for i in cols_to_drop: #Loop through the columns to drop
                logging.info(f'{i} dropped with {missing_perc[i]:.2%} of missing values')
            logging.info(f'{len(cols_to_drop)} columns dropped due to incompleteness')
            return df.loc[:, missing_perc < thres]

    def impute_missings(self, df):
        '''
        Function imputing missing values
        '''
        #Get columns with missing values
        miss_cols = df.columns[df.count() < df.shape[0]].to_list()
        if miss_cols == []:
            logging.info('No values were imputed')
            return df

        #Impute discrete variables with mode
        mode_cols = df[miss_cols].nunique() <= 20

        