### NOTES #####################################################################

#Uses environment 'Reserving-RNN-PyTorch'
#python 3.10.13
#pytorch 2.2.1
#torchvision 0.17.1
#cuda 12.1.0
#pandas 2.1.1
#numpy 1.26.0

# Takes simulated SPLICE output from R, converts it into a 'set' and 'index' 
# csv file, splits into training and test sets and stores them in csvs

# Only difference between this an 'Data Manipulation.py' is that this file
# splits the data by both claim and by time

### PACKAGES ##################################################################

import pandas as pd
import numpy as np
from random import sample, seed
import pathlib

### FILEPATHS (ADJUST THESE BEFORE RUNNING) ###################################

# filepaths noInf
fp_in = 'Datasets/R Outputs/data_noInf_cov_TRUE_seed_20201006.csv'

#fp_in = 'Datasets/R Outputs/test_incurred_dataset_noInf.csv'

fp_out_v1 = 'Datasets/Python Inputs/V1/noInf Triangular WithInc/'
fp_out_v1_noInc = 'Datasets/Python Inputs/V1/noInf Triangular NoInc/'


### FUNCTIONS #################################################################
def get_ultimate(row):
    '''Finds the ultimate claim cost for the given claim for a particular row'''
    return data.loc[data['claim_no'] == row['claim_no'], ['cumpaid']].iloc[-1]

def get_incurred(row):
    '''Finds the estimated incurred claim cost for the given claim for a 
    particular row'''
    return data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] < row['c']), ['incurred']].iloc[-1]

def get_cumpaid(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c']), ['cumpaid']].max()

def get_case_estimate(row):
    temp = data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] <= row['c'])]
    
    return temp.loc[(temp['txn_time'] == max(temp['txn_time'])), 
                    'incurred'].iloc[0]

def get_finalised_quarter(row):
    return data.loc[data['claim_no'] == row['claim_no'], ['txn_time']].iloc[-1]

def get_legal_rep(row):
    return data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] < row['c']), ['Legal Representation']].iloc[-1]

def get_injury_sev(row):
    return data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] < row['c']), ['Injury Severity']].iloc[-1]
    
def get_age(row):
    return data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] < row['c']), ['Age of Claimant']].iloc[-1]

### DATA MANIPULATION #########################################################

# Reading data
data = pd.read_csv(fp_in)

#data.head()

# Dropping first column
data = data.iloc[:, 1:]

# Adding duration and start time columns, and rounding them
start_group = data.groupby('claim_no')['txn_time'].min()
start_group = np.ceil(start_group)

end_group = data.groupby('claim_no')['txn_time'].max()
end_group = np.ceil(end_group)

# Adding raw payments to old dataframe
payments = data['cumpaid'].diff().fillna(0)
payments[payments < 0] = 0
data['payments'] = payments

# Adding finalisation time to old dataframe
data['finalised_quarter'] = np.ceil(data.apply(get_finalised_quarter, axis = 1))

# Adding censoring times to index dataframe
index_data_list = []

for claim_no in range(1, len(start_group) + 1):
    for c in range(int(start_group[claim_no]), int(end_group[claim_no])):
        dev_quarter = c - start_group[claim_no] + 1
        occ_quarter = start_group[claim_no]
        finalised_quarter = data.loc[data['claim_no'] == claim_no, 'finalised_quarter'].mean()
        index_data_list.append([claim_no, c, dev_quarter, occ_quarter, finalised_quarter])
        
index_data = pd.DataFrame(index_data_list, columns = ['claim_no', 'c', 
                                                      'dev_quarter', 
                                                      'occ_quarter', 
                                                      'finalised_quarter'])       

# Extracting claim size, incurred and current case estimate values
index_data['claim_size'] = index_data.apply(get_ultimate, axis = 1)
index_data['incurred'] = index_data.apply(get_incurred, axis = 1)
index_data['case_estimate'] = index_data.apply(get_case_estimate, axis = 1)

index_data['Legal Representation'] = index_data.apply(get_legal_rep, axis = 1)
index_data['Injury Severity'] = index_data.apply(get_injury_sev, axis = 1)
index_data['Age of Claimant'] = index_data.apply(get_age, axis = 1)

# Finding rows in original dataset that involve payments
payment_rows = data.loc[(data['txn_type'] == 'P') | 
                        (data['txn_type'] == 'PMi') | 
                        (data['txn_type'] == 'PMa')]

# Adding payment data
index_data['cumpaid'] = index_data.apply(get_cumpaid, axis = 1)

# Adding indexes to index set
index_data['index'] = index_data.index

# Adding m(t) and log(m(t)) to index dataset
index_data['m'] = index_data['claim_size'] / index_data['case_estimate']
index_data['log_m'] = np.log(index_data['m'])

# Creating dataframe with only censored rows
databoxes_list = []

# Loop over indexes
for i in range(len(index_data)):

    # subset claims from original dataset
    temp = data.loc[data['claim_no'] == index_data.iloc[i]['claim_no'], 
                    ['claim_no', 'txn_time', 'txn_delay', 'txn_type', 
                     'incurred', 'OCL', 'cumpaid']]

    # Loop over rows in original dataset
    for j in range(len(temp)):
        if (index_data.iloc[i]['claim_no'] == temp.iloc[j]['claim_no'] and 
            index_data.iloc[i]['c'] > temp.iloc[j]['txn_time']):

            databoxes_list.append([i] + 
                                  [index_data.iloc[i]['c']] + 
                                  list(temp.iloc[j]))

# converting list to dataframe
databoxes = pd.DataFrame(databoxes_list, columns = ['index', 'c', 'claim_no', 
                                                    'txn_time', 'txn_delay', 
                                                    'txn_type', 'incurred', 
                                                    'OCL', 'cumpaid'])

index_data['cumpaid'] = index_data['cumpaid'].fillna(0)

# adding true_ocl to index data
index_data['true_ocl'] = index_data['claim_size'] - index_data['cumpaid']
index_data['log_true_ocl'] = np.log(index_data['true_ocl'])

# renaming columns
index_data.rename(columns = {'c': 'pred_time'}, inplace = True)
databoxes.rename(columns = {'c': 'pred_time', 'txn_time': 'cal_time', 
                            'txn_delay': 'dev_time'}, inplace = True)

# Replacing NAs with -1
index_data = index_data.fillna(-1)
databoxes = databoxes.fillna(-1)

# rename columns
databoxes.rename(columns = {'cumpaid': 'paid', 'OCL': 'ocl'}, inplace = True)
index_data.rename(columns = {'incurred': 'latest_incurred'}, inplace = True)

# Some claim reports have payments really close to 0, need to set these to exactly 0
databoxes.loc[databoxes['paid'] < 0.01, 'paid'] = 0

# Binary encode legal rep covariate
index_data['Legal Representation'].replace({'Y': 1, 'N': 0}, inplace=True)

# Ordinal encode age covariate
index_data['Age of Claimant'].replace({'0-15': 0, '15-30': 2, '30-50': 2, '50-65': 3, 'over 65': 4}, inplace=True)

# Make injury severity go from 0-5 instead of 1-6 so it works with embeddings
index_data['Injury Severity'] = index_data['Injury Severity'] - 1

### TRAIN TEST SPLIT ##########################################################

# Valuation date is 40
train_index = index_data.loc[index_data['finalised_quarter'] <= 36]
val_index = index_data.loc[(index_data['finalised_quarter'] > 36) & (index_data['finalised_quarter'] <= 40)]
test_index = index_data.loc[index_data['finalised_quarter'] > 40]

train_set = databoxes.loc[databoxes['claim_no'].isin(train_index['claim_no'])]
val_set = databoxes.loc[databoxes['claim_no'].isin(val_index['claim_no'])]
test_set = databoxes.loc[databoxes['claim_no'].isin(test_index['claim_no'])]

# creating noIncurred versions
train_set_noInc = train_set.loc[((train_set['txn_type'] != 'Ma') | (train_set['paid'] == 0)) & (train_set['txn_type'] != 'Mi')]
val_set_noInc = val_set.loc[((val_set['txn_type'] != 'Ma') | (val_set['paid'] == 0)) & (val_set['txn_type'] != 'Mi')]
test_set_noInc = test_set.loc[((test_set['txn_type'] != 'Ma') | (test_set['paid'] == 0)) & (test_set['txn_type'] != 'Mi')]

# creating different sets for different model input versions

v1_train_set = train_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                                 'cal_time', 'paid', 'ocl']]

v1_val_set = val_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                             'cal_time', 'paid', 'ocl']]

v1_test_set = test_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                               'cal_time', 'paid', 'ocl']]

v1_train_index = train_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                     'dev_quarter', 'occ_quarter', 
                                     'finalised_quarter', 'claim_size', 
                                     'latest_incurred', 'm', 'log_m',
                                     'true_ocl', 'log_true_ocl',
                                     'Legal Representation', 'Injury Severity',
                                     'Age of Claimant']]

v1_val_index = val_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                 'dev_quarter', 'occ_quarter', 
                                 'finalised_quarter', 'claim_size', 
                                 'latest_incurred', 'm', 'log_m',
                                 'true_ocl', 'log_true_ocl',
                                 'Legal Representation', 'Injury Severity',
                                 'Age of Claimant']]

v1_test_index = test_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                   'dev_quarter', 'occ_quarter', 
                                   'finalised_quarter', 'claim_size', 
                                   'latest_incurred', 'm', 'log_m',
                                   'true_ocl', 'log_true_ocl',
                                   'Legal Representation', 'Injury Severity',
                                   'Age of Claimant']]

v1_train_set_noInc = train_set_noInc.loc[:, ['index', 'claim_no', 'pred_time', 
                                             'dev_time', 'cal_time', 'paid', 
                                             'ocl']]

v1_val_set_noInc = val_set_noInc.loc[:, ['index', 'claim_no', 'pred_time', 
                                         'dev_time', 'cal_time', 'paid', 'ocl']]

v1_test_set_noInc = test_set_noInc.loc[:, ['index', 'claim_no', 'pred_time', 
                                     'dev_time', 'cal_time', 'paid', 'ocl']]


### EXPORTING #################################################################
# V1
v1_train_index.to_csv(fp_out_v1 + 'train_index.csv', index = False)
v1_val_index.to_csv(fp_out_v1 + 'val_index.csv', index = False)
v1_test_index.to_csv(fp_out_v1 + 'test_index.csv', index = False)

v1_train_set.to_csv(fp_out_v1 + 'train_set.csv', index = False)
v1_val_set.to_csv(fp_out_v1 + 'val_set.csv', index = False)
v1_test_set.to_csv(fp_out_v1 + 'test_set.csv', index = False)

# No Incurred
v1_train_index.to_csv(fp_out_v1_noInc + 'train_index.csv', index = False)
v1_val_index.to_csv(fp_out_v1_noInc + 'val_index.csv', index = False)
v1_test_index.to_csv(fp_out_v1_noInc + 'test_index.csv', index = False)

v1_train_set_noInc.to_csv(fp_out_v1_noInc + 'train_set.csv', index = False)
v1_val_set_noInc.to_csv(fp_out_v1_noInc + 'val_set.csv', index = False)
v1_test_set_noInc.to_csv(fp_out_v1_noInc + 'test_set.csv', index = False)

# checking max sequence length (whether to increase the limit for the model)
#databoxes.groupby('index').count().sort_values('claim_no', ascending = False)