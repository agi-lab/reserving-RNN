### NOTES #####################################################################

#Uses environment 'Reserving-RNN-PyTorch'
#python 3.10.13
#pytorch 2.2.1
#torchvision 0.17.1
#cuda 12.1.0
#pandas 2.1.1
#numpy 1.26.0

#Takes simulated SPLICE output from R, converts it into a 'set' and 'index' 
# csv file, splits into training and test sets and stores them in csvs

#For each R output, creates files of versions V1 (cal_time, dev_time, cumpaid, 
# ocl), V2 (V1 + txn_type, multiplier) and V3 (V2 into RNN layers + rest of 
# derived features into FC layer)

# Only difference between this an 'Data Manipulation.py' is that this file
# splits the data by both claim and by time

### PACKAGES ##################################################################

import pandas as pd
import numpy as np
from random import sample, seed

### FILEPATHS (ADJUST THESE BEFORE RUNNING) ###################################

# filepaths noInf
fp_in = 'Datasets/R Outputs/test_incurred_dataset_noInf.csv'
fp_out_v1 = 'Datasets/Python Inputs/V1/noInf Time/'
fp_out_v2 = 'Datasets/Python Inputs/V2/noInf Time/'
fp_out_v3 = 'Datasets/Python Inputs/V3/noInf Time/'

# filepaths Inflated
#fp_in = 'Datasets/R Outputs/test_incurred_dataset_Inflated.csv'
#fp_out_v1 = 'Datasets/Python Inputs/V1/Inflated Time/'
#fp_out_v2 = 'Datasets/Python Inputs/V2/Inflated Time/'
#fp_out_v3 = 'Datasets/Python Inputs/V3/Inflated Time/'


### FUNCTIONS #################################################################
def get_ultimate(row):
    '''Finds the ultimate claim cost for the given claim for a particular row'''
    return data.loc[data['claim_no'] == row['claim_no'], ['cumpaid']].max()

def get_incurred(row):
    '''Finds the estimated incurred claim cost for the given claim for a 
    particular row'''
    return data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] < row['c']), ['incurred']].iloc[-1]

def get_num_payments(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c'])].shape[0]

def get_mean_payments(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c'])]['payments'].mean()

def get_var_payments(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c'])]['payments'].var()

def get_max_payments(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c'])]['payments'].max()

def get_cumpaid(row):
    return payment_rows.loc[(payment_rows['claim_no'] == row['claim_no']) & 
                            (payment_rows['txn_time'] <= row['c'])]['cumpaid'].max()

def get_case_estimate(row):
    temp = data.loc[(data['claim_no'] == row['claim_no']) & 
                    (data['txn_time'] <= row['c'])]
    
    return temp.loc[(temp['txn_time'] == max(temp['txn_time'])), 
                    'incurred'].iloc[0]

def get_num_revisions(row):
    return revision_rows.loc[(revision_rows['claim_no'] == row['claim_no']) & 
                             (revision_rows['txn_time'] <= row['c'])].shape[0]

def get_upward_revisions(row):
    temp = revision_rows.loc[(revision_rows['claim_no'] == row['claim_no']) & 
                             (revision_rows['txn_time'] <= row['c'])]
    
    temp['revision'] = temp['incurred'].diff()
    return temp.loc[temp['revision'] > 0].shape[0]

def get_total_variation(row):
    temp = revision_rows.loc[(revision_rows['claim_no'] == row['claim_no']) & 
                             (revision_rows['txn_time'] <= row['c'])]
    
    init_est = temp.loc[(temp['txn_time'] == min(temp['txn_time'])), 
                        'incurred'].iloc[0]
    
    cur_est = temp.loc[(temp['txn_time'] == max(temp['txn_time'])), 
                       'incurred'].iloc[0]
    
    return cur_est - init_est

def get_finalised_quarter(row):
    return data.loc[data['claim_no'] == row['claim_no'], 'txn_time'].max()

### DATA MANIPULATION #########################################################

# Reading data
data = pd.read_csv(fp_in)

data.head()

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


# testing
'''data['finalised_quarter'].max()

plt.hist(data['finalised_quarter'])
plt.show()

plt.hist(data.groupby('claim_no').last()['finalised_quarter'])
plt.show()

print(f"# Claims finalised after quarter 48: {sum(data.groupby('claim_no').last()['finalised_quarter'] > 48)}")
print(f"# Claims finalised before and including quarter 48: {sum(data.groupby('claim_no').last()['finalised_quarter'] <= 48)}")

print(f" prop Claims finalised after quarter 48: {np.mean(data.groupby('claim_no').last()['finalised_quarter'] > 48)}")
print(f" prop Claims finalised between quarters 40 and 48: {np.mean((data.groupby('claim_no').last()['finalised_quarter'] > 40) & (data.groupby('claim_no').last()['finalised_quarter'] <= 48))}")
print(f" prop Claims finalised before and including quarter 48: {np.mean(data.groupby('claim_no').last()['finalised_quarter'] <= 40)}")
'''

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

# Finding rows in original dataset that involve payments
payment_rows = data.loc[(data['txn_type'] == 'P') | 
                        (data['txn_type'] == 'PMi') | 
                        (data['txn_type'] == 'PMa')]

# Adding payment data
index_data['num_payments'] = index_data.apply(get_num_payments, axis = 1)
index_data['mean_payments'] = index_data.apply(get_mean_payments, axis = 1)
index_data['var_payments'] = index_data.apply(get_var_payments, axis = 1)
index_data['max_payment'] = index_data.apply(get_max_payments, axis = 1)
index_data['cumpaid'] = index_data.apply(get_cumpaid, axis = 1)


# Finding rows in original dataset that involve revisions
revision_rows = data.loc[(data['txn_type'] != 'P')]

# Adding revisions data
index_data['num_revisions'] = index_data.apply(get_num_revisions, axis = 1)
index_data['num_upward'] = index_data.apply(get_upward_revisions, axis = 1)
index_data['total_variation'] = index_data.apply(get_total_variation, axis = 1)

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
                     'incurred', 'OCL', 'cumpaid', 'multiplier']]

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
                                                    'OCL', 'cumpaid', 
                                                    'multiplier'])


# renaming columns
index_data.rename(columns = {'c': 'pred_time'}, inplace = True)
databoxes.rename(columns = {'c': 'pred_time', 'txn_time': 'cal_time', 
                            'txn_delay': 'dev_time'}, inplace = True)

# Replacing NAs with -1
index_data = index_data.fillna(-1)
databoxes = databoxes.fillna(-1)


# converting txn_type into numeric features (is_payment, is_major, is_minor)
databoxes['is_payment'] = [1 if (element == 'P' or 
                                 element == 'PMi' or 
                                 element == 'PMa') else 0 
                                 for element in databoxes['txn_type']]

databoxes['is_major'] = [1 if (element == 'Ma' or element == 'PMa') else 0 
                         for element in databoxes['txn_type']]
databoxes['is_minor'] = [1 if (element == 'Mi' or element == 'PMi') else 0 
                         for element in databoxes['txn_type']]

# rename columns
databoxes.rename(columns = {'cumpaid': 'paid', 'OCL': 'ocl'}, inplace = True)
index_data.rename(columns = {'claim_size': 'target', 
                             'incurred': 'latest_incurred'}, inplace = True)


# observation level
import matplotlib.pyplot as plt
plt.hist(index_data['finalised_quarter'], bins = 20)
plt.show()

print(f"max occurrence quarter: {index_data['occ_quarter'].max()}")
print(f"max finalisation quarter: {index_data['finalised_quarter'].max()}")

print(f"# Observations finalised after quarter 48: {sum(index_data['finalised_quarter'] > 48)}")
print(f"# Observations finalised between quarters 42 and 48: {sum((index_data['finalised_quarter'] > 42) & (index_data['finalised_quarter'] <= 48))}")
print(f"# Observations finalised before and including quarter 42: {sum(index_data['finalised_quarter'] <= 42)}")

print(f"prop Observations finalised after quarter 48: {np.mean(index_data['finalised_quarter'] > 48):.3f}")
print(f"# Observations finalised between quarters 42 and 48: {np.mean((index_data['finalised_quarter'] > 42) & (index_data['finalised_quarter'] <= 48)):.3f}")
print(f"prop Observations finalised before and including quarter 42: {np.mean(index_data['finalised_quarter'] <= 42):.3f}")

# claim level
plt.hist(index_data.groupby('claim_no').last()['finalised_quarter'])
plt.show()

print(f"# Claims finalised after quarter 48: {sum(index_data.groupby('claim_no').last()['finalised_quarter'] > 48)}")
print(f"# Claims finalised between quarters 42 and 48: {sum((index_data.groupby('claim_no').last()['finalised_quarter'] > 42) & (index_data.groupby('claim_no').last()['finalised_quarter'] <= 48))}")
print(f"# Claims finalised before and including quarter 48: {sum(index_data.groupby('claim_no').last()['finalised_quarter'] <= 48)}")

print(f" prop Claims finalised after quarter 48: {np.mean(index_data.groupby('claim_no').last()['finalised_quarter'] > 48):.3f}")
print(f" prop Claims finalised between quarters 42 and 48: {np.mean((index_data.groupby('claim_no').last()['finalised_quarter'] > 42) & (index_data.groupby('claim_no').last()['finalised_quarter'] <= 48)):.3f}")
print(f" prop Claims finalised before and including quarter 42: {np.mean(index_data.groupby('claim_no').last()['finalised_quarter'] <= 42):.3f}")


# checking proportions at cal time 40 (this should be true valuation date)
print(f"# Observations finalised after quarter 40: {sum(index_data['finalised_quarter'] > 40)}")
print(f"# Observations finalised between quarters 36 and 40: {sum((index_data['finalised_quarter'] > 36) & (index_data['finalised_quarter'] <= 40))}")
print(f"# Observations finalised before and including quarter 36: {sum(index_data['finalised_quarter'] <= 36)}")

print(f"prop Observations finalised after quarter 40: {np.mean(index_data['finalised_quarter'] > 40):.3f}")
print(f"# Observations finalised between quarters 36 and 40: {np.mean((index_data['finalised_quarter'] > 36) & (index_data['finalised_quarter'] <= 40)):.3f}")
print(f"prop Observations finalised before and including quarter 36: {np.mean(index_data['finalised_quarter'] <= 36):.3f}")

print(f"# Claims finalised after quarter 40: {sum(index_data.groupby('claim_no').last()['finalised_quarter'] > 40)}")
print(f"# Claims finalised between quarters 36 and 40: {sum((index_data.groupby('claim_no').last()['finalised_quarter'] > 36) & (index_data.groupby('claim_no').last()['finalised_quarter'] <= 40))}")
print(f"# Claims finalised before and including quarter 40: {sum(index_data.groupby('claim_no').last()['finalised_quarter'] <= 40)}")

print(f" prop Claims finalised after quarter 40: {np.mean(index_data.groupby('claim_no').last()['finalised_quarter'] > 40):.3f}")
print(f" prop Claims finalised between quarters 36 and 40: {np.mean((index_data.groupby('claim_no').last()['finalised_quarter'] > 36) & (index_data.groupby('claim_no').last()['finalised_quarter'] <= 40)):.3f}")
print(f" prop Claims finalised before and including quarter 36: {np.mean(index_data.groupby('claim_no').last()['finalised_quarter'] <= 36):.3f}")


### TRAIN TEST SPLIT ##########################################################

# Valuation date is 48
#train_index = index_data.loc[index_data['finalised_quarter'] <= 42]
#val_index = index_data.loc[(index_data['finalised_quarter'] > 42) & (index_data['finalised_quarter'] <= 48)]
#test_index = index_data.loc[index_data['finalised_quarter'] > 48]

#train_set = databoxes.loc[databoxes['claim_no'].isin(train_index['claim_no'])]
#val_set = databoxes.loc[databoxes['claim_no'].isin(val_index['claim_no'])]
#test_set = databoxes.loc[databoxes['claim_no'].isin(test_index['claim_no'])]

# Valuation date is 40
train_index = index_data.loc[index_data['finalised_quarter'] <= 36]
val_index = index_data.loc[(index_data['finalised_quarter'] > 36) & (index_data['finalised_quarter'] <= 40)]
test_index = index_data.loc[index_data['finalised_quarter'] > 40]

train_set = databoxes.loc[databoxes['claim_no'].isin(train_index['claim_no'])]
val_set = databoxes.loc[databoxes['claim_no'].isin(val_index['claim_no'])]
test_set = databoxes.loc[databoxes['claim_no'].isin(test_index['claim_no'])]


### EXPORTING #################################################################

# creating different sets for different model input versions

v1_train_set = train_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                                 'cal_time', 'paid', 'ocl']]

v1_val_set = val_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                             'cal_time', 'paid', 'ocl']]

v1_test_set = test_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                               'cal_time', 'paid', 'ocl']]

v2_train_set = train_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                                 'cal_time', 'paid', 'ocl', 'is_payment', 
                                 'is_major', 'is_minor', 'multiplier']]

v2_val_set = val_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                             'cal_time', 'paid', 'ocl', 'is_payment', 
                             'is_major', 'is_minor', 'multiplier']]

v2_test_set = test_set.loc[:, ['index', 'claim_no', 'pred_time', 'dev_time', 
                               'cal_time', 'paid', 'ocl', 'is_payment', 
                               'is_major', 'is_minor', 'multiplier']]

v1_train_index = train_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                     'dev_quarter', 'occ_quarter', 
                                     'finalised_quarter', 'target', 
                                     'latest_incurred', 'm', 'log_m']]

v1_val_index = val_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                 'dev_quarter', 'occ_quarter', 
                                 'finalised_quarter', 'target', 
                                 'latest_incurred', 'm', 'log_m']]

v1_test_index = test_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                   'dev_quarter', 'occ_quarter', 
                                   'finalised_quarter', 'target', 
                                   'latest_incurred', 'm', 'log_m']]

v3_train_index = train_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                     'dev_quarter', 'occ_quarter', 
                                     'finalised_quarter', 'target', 
                                     'latest_incurred', 'm', 'log_m', 
                                     'num_payments', 'mean_payments', 
                                     'var_payments', 'max_payment', 
                                     'num_revisions', 'num_upward', 
                                     'total_variation']]

v3_val_index = val_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                 'dev_quarter', 'occ_quarter', 
                                 'finalised_quarter', 'target', 
                                 'latest_incurred', 'm', 'log_m', 
                                 'num_payments', 'mean_payments', 
                                 'var_payments', 'max_payment',
                                   'num_revisions', 'num_upward', 
                                   'total_variation']]

v3_test_index = test_index.loc[:, ['index', 'claim_no', 'pred_time', 
                                   'dev_quarter', 'occ_quarter', 
                                   'finalised_quarter', 'target', 
                                   'latest_incurred', 'm', 'log_m', 
                                   'num_payments', 'mean_payments', 
                                   'var_payments', 'max_payment', 
                                   'num_revisions', 'num_upward', 
                                   'total_variation']]


# Exporting to CSVs

# V1
v1_train_index.to_csv(fp_out_v1 + 'train_index.csv', index = False)
v1_val_index.to_csv(fp_out_v1 + 'val_index.csv', index = False)
v1_test_index.to_csv(fp_out_v1 + 'test_index.csv', index = False)

v1_train_set.to_csv(fp_out_v1 + 'train_set.csv', index = False)
v1_val_set.to_csv(fp_out_v1 + 'val_set.csv', index = False)
v1_test_set.to_csv(fp_out_v1 + 'test_set.csv', index = False)

# V2
v1_train_index.to_csv(fp_out_v2 + 'train_index.csv', index = False)
v1_val_index.to_csv(fp_out_v2 + 'val_index.csv', index = False)
v1_test_index.to_csv(fp_out_v2 + 'test_index.csv', index = False)

v2_train_set.to_csv(fp_out_v2 + 'train_set.csv', index = False)
v2_val_set.to_csv(fp_out_v2 + 'val_set.csv', index = False)
v2_test_set.to_csv(fp_out_v2 + 'test_set.csv', index = False)

# V3
v3_train_index.to_csv(fp_out_v3 + 'train_index.csv', index = False)
v3_val_index.to_csv(fp_out_v3 + 'val_index.csv', index = False)
v3_test_index.to_csv(fp_out_v3 + 'test_index.csv', index = False)

v2_train_set.to_csv(fp_out_v3 + 'train_set.csv', index = False)
v2_val_set.to_csv(fp_out_v3 + 'val_set.csv', index = False)
v2_test_set.to_csv(fp_out_v3 + 'test_set.csv', index = False)
