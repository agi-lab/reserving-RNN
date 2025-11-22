### IMPORTS ###################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_

# to use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### HELPER FUNCTIONS/CLASSES ##################################################

def save_dictionary(data, file_path):
    """Saves a dictionary as a pickle file.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The path to the folder where the dictionary will be saved.
    """
    full_path = os.path.join(file_path, 'hyperparameter_grid.pickle')

    with open(full_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dictionary(file_path):
    """Loads a dictionary from a pickle file.

    Args:
        file_path (str): The path to the file from which the dictionary will be loaded.

    Returns:
        dict: The loaded dictionary, or None if an error occurs.
    """

    full_path = os.path.join(file_path, 'hyperparameter_grid.pickle')

    if not os.path.exists(full_path):
         print(f"Error: File not found: {full_path}")
         return None
    
    with open(full_path, 'rb') as handle:
        try:
            data = pickle.load(handle)
            return data
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return None

def round_threshold(num, threshold=1000):
    """Takes a float and rounds it to the nearest integer if it is above a 
    certain threshold. Otherwise rounds the float to 3 decimal places."""

    if num >= threshold:
        return int(num)
    
    else:
        return round(num, 3)
    
# errors as defined in the essay
# could be used in model training, currently only used for diagnostics

class MeanAbsoluteLogError(nn.Module):
    def __init__(self):
        super(MeanAbsoluteLogError, self).__init__()

    def forward(self, preds, actuals):

        # convert negative preds to 1 so that the log will be 0
        # smallest true OCL is 16 (training set) 7000 (test set) so penalty will be severe as a test set metric
        preds_copy = preds.copy()
        preds_copy[preds_copy < 1] = 1

        if torch.is_tensor(preds_copy):
            return torch.mean(torch.abs(torch.log(preds_copy) - torch.log(actuals)))
        else:
            return np.mean(np.abs(np.log(preds_copy) - np.log(actuals)))
    
class MeanSquaredLogError(nn.Module):
    def __init__(self):
        super(MeanSquaredLogError, self).__init__()

    def forward(self, preds, actuals):

        # convert negative preds to 1 so that the log will be 0
        # smallest true OCL is 16 (training set) 7000 (test set) so penalty will be severe as a test set metric
        preds_copy = preds.copy()
        preds_copy[preds_copy < 1] = 1

        if torch.is_tensor(preds_copy):
            return torch.mean(torch.square(torch.log(preds_copy) - torch.log(actuals)))
        else:
            return np.mean(np.square(np.log(preds_copy) - np.log(actuals)))
    
class MSLE_with_penalty(nn.Module):
    '''Testing a new loss function. Works the same as the regular MSE but adds 
    a penalty for negative OCL predictions. Should only be used with models 
    that predict claim_size or log_m.'''

    def __init__(self, pen_weight=1, pen_type='constant'):
        super(MSLE_with_penalty, self).__init__()
        self.pen_weight = pen_weight
        self.pen_type = pen_type # 'constant', 'linear' or 'log'

    def forward(self, raw_preds, targets, lower_bounds, preds):
        # lower bounds refers to the cumulative payments to date (ultimate claim size cannot be less than this)
        # raw_preds are in terms of the model's output (e.g. log_m), preds are transformed to always be in terms of ultimate claim size

        msle = torch.nn.MSELoss()(raw_preds, targets) # raw preds and targets are already in terms of log_m

        if self.pen_type == 'constant':
            penalty = self.pen_weight * torch.sum(lower_bounds > preds)
        
        elif self.pen_type == 'linear':
            penalty = self.pen_weight * torch.mean(torch.maximum(torch.zeros_like(preds), lower_bounds - preds))
        
        elif self.pen_type == 'log':
            penalty = self.pen_weight * torch.mean(torch.maximum(torch.zeros_like(preds), torch.log(lower_bounds) - torch.log(preds)))

        else:
            raise ValueError("pen_type must be 'constant', 'linear' or 'log'")

        return msle + penalty

    def __repr__(self):
        return f"MSLE_with_penalty(pen_weight={self.pen_weight}, pen_type={self.pen_type})"

    def __str__(self):
        return self.__repr__()

def initialise_weights(model):
    """
    Initialize weights of a PyTorch model, including handling RNNs, LSTMs, GRUs, Linear, 
    BatchNorm, and LayerNorm layers.

    - RNN/LSTM/GRU weights: Kaiming Normal
    - Linear weights: Kaiming Normal
    - BatchNorm/LayerNorm weights: Gamma (scale) = 1.0, Beta (shift) = 0.0
    """
    for name, param in model.named_parameters():
        # Handle RNNs, LSTMs, and GRUs
        if 'weight_ih' in name:  # Input-to-hidden weights
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        elif 'weight_hh' in name:  # Hidden-to-hidden weights
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        elif 'bias' in name:  # Bias terms
            nn.init.zeros_(param)

    # Handle all modules explicitly
    for module in model.modules():
        if isinstance(module, nn.Linear):  # Linear layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):  # BatchNorm
            if module.weight is not None:  # Gamma
                nn.init.ones_(module.weight)
            if module.bias is not None:  # Beta
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):  # LayerNorm
            if module.weight is not None:  # Gamma
                nn.init.ones_(module.weight)
            if module.bias is not None:  # Beta
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    # LSTM: [input, forget, cell, output]
                    bias_size = param.size(0)
                    gate_size = bias_size // 4
                    with torch.no_grad():
                        param[gate_size:2*gate_size].fill_(1.0)  # forget gate
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    # GRU: [reset, update, new]
                    bias_size = param.size(0)
                    gate_size = bias_size // 3
                    with torch.no_grad():
                        param[gate_size:2*gate_size].fill_(1.0)  # update gate (or choose as needed)

def create_grid(target_cols, criterions, types, output_layers, 
                nOuts, epochss, nHiddens, nLayerss, patiences, batch_sizes, 
                optimisers, lrs, nonlinearitys, dropouts, normalisations, 
                include_incurredss, include_covariatess, transform_inputss, model_types):
    
    '''
    Inputs: lists of hyperparameter values
    Output: a list of dictionaries to be used in the cross_validate function'''

    hyperparameter_grid = []

    for params in product(target_cols, criterions, types, 
                          output_layers, nOuts, epochss, nHiddens, nLayerss, 
                          patiences, batch_sizes, optimisers, lrs, nonlinearitys, 
                          dropouts, normalisations, include_incurredss, 
                          include_covariatess, transform_inputss, model_types):
        
        hyperparameter_grid.append({
            'target_col': params[0],
            'criterion': params[1],
            'type': params[2],
            'output_layer': params[3],
            'nOut': params[4],
            'epochs': params[5],
            'nHidden': params[6],
            'nLayers': params[7],
            'patience': params[8],
            'batch_size': params[9],
            'optimiser': params[10],
            'lr': params[11],
            'nonlinearity': params[12],
            'dropout': params[13],
            'normalisation': params[14],
            'include_incurreds': params[15],
            'include_covariates': params[16],
            'transform_inputs': params[17],
            'model_type': params[18]
        })

    return hyperparameter_grid

median_colours = {'LSTM+': "#5D3EF8",
                  'LSTM': "#0DA1EB",
                  'FNN+': "#F147D5",
                  'FNN': "#07BB43",
                  'Case Estimates': "#D4B206"}

edge_colours = {'LSTM+': '#332288',
                'LSTM': "#3B8BB3",
                'FNN+': '#AA4499',
                'FNN': '#117733',
                'Case Estimates': "#B19A27"}

fill_colours = {'LSTM+': "#B1A2FC",
                'LSTM': '#88CCEE',
                'FNN+': "#F8B9EE",
                'FNN': "#B3F7CA",
                'Case Estimates': "#F1E296"}

def get_median_colour(model_name):
    if model_name in median_colours.keys():
        return median_colours[model_name]
    return "#FF0000"

def get_edge_colour(model_name):
    if model_name in edge_colours.keys():
        return edge_colours[model_name]
    return "#FF0000"

def get_fill_colour(model_name):
    if model_name in fill_colours.keys():
        return fill_colours[model_name]
    return "#FF0000"

def box_plot(data, positions, model_name=None, alpha=1, widths=0.7, showfliers=False):
    data = pd.DataFrame(data)

    # dataframe boxplot needed over matplotlib or seaborn to handle missing values correctly !!!
    bp = data.boxplot(backend='matplotlib', return_type='dict', grid=False, positions=positions, widths=widths, patch_artist=True, showfliers=showfliers)

    plt.setp(bp['medians'], color=get_median_colour(model_name), linewidth=2)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        plt.setp(bp[element], color=get_edge_colour(model_name), alpha=alpha)

    for patch in bp['boxes']:
        patch.set(facecolor=get_fill_colour(model_name), alpha=alpha)
        
    return bp

def bias_correction_factor(preds, targets):
    ''' assumes preds and targets are both on actual scale (i.e. not log scale) '''

    # non-parametric
    bias = np.mean(np.exp(np.log(targets) - np.log(preds)))

    # log-normal assumption
    #bias = np.exp(0.5 * np.var(np.log(targets) - np.log(preds)))

    print(f'Bias correction factor: {bias:.3f}')
    return bias

def get_model_params_num(model, trainable_only=False):
    """
    Calculates the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        trainable_only (bool, optional): If True, only count trainable parameters. Defaults to False.

    Returns:
        int: The total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

### MODEL CLASSES #############################################################

class ClaimsDataset(Dataset):
    """ Based on Arkie's ClaimsDataset
    
    Notes: 
    - dataloader has to iterate from 0:len(dataset)
    - all sequences are padded to a minimum length of 70
    """

    def __init__(self, target_col, index_path, set_path, include_incurreds=True, 
                 include_covariates=False, transform_inputs=False, model_type='RNN', scaler=None):
        self.target_col = target_col # string referring to name of the target column (i.e. 'claim_size', 'log_claim_size', 'log_m', 'true_ocl' or 'log_true_ocl')
        self.index = pd.read_csv(index_path) 
        self.set = pd.read_csv(set_path)
        self.include_incurreds = include_incurreds # boolean whether to use case estimate data or not
        self.include_covariates = include_covariates # boolean whether to include covariate data or not
        self.transform_inputs = transform_inputs # boolean whether to transform inputs or not
        self.model_type = model_type # string referring to the type of model being used (either 'RNN' (includes LSTM and GRU) or 'FNN')
        self.scaler = scaler # dictionary of scalers to be applied to the data

        # creating new column for target output (so that any scalings applied to the target column are not applied to the original data)
        self.index['target'] = self.index[self.target_col]
        self.index['incurred_copy'] = self.index['latest_incurred']
        self.index['pred_time_copy'] = self.index['pred_time']
        self.index['acc_quarter_copy'] = self.index['acc_quarter']

        if self.transform_inputs:

            # log transform some inputs
            self.set['paid'] = np.log(self.set['paid'] + 1)
            self.set['dev_time'] = np.log(self.set['dev_time'] + 1)
            if self.include_incurreds:
                self.set['ocl'] = np.log(self.set['ocl'] + 1)

            if self.model_type == 'FNN':
                self.index['mean_payments'] = np.log(self.index['mean_payments'] + 1)
                self.index['vco_payments'] = np.log(self.index['vco_payments'] + 1)
                self.index['max_payment'] = np.log(self.index['max_payment'] + 1)

                if self.include_incurreds:
                    self.index['total_revisions'] = np.log(self.index['total_revisions'] + 1)
                    self.index['incurred_copy'] = np.log(self.index['incurred_copy'] + 1)

            # standardise inputs and output
            if self.scaler is None:
                # learn scalings and apply them
                self.scaler = {'target': StandardScaler(), 
                               'paid': StandardScaler(), 
                               'ocl': StandardScaler(), 
                               'dev_time': StandardScaler(), 
                               'cal_time': StandardScaler(),
                               # can't scale the two below because it will interfere with analysis by time
                               'pred_time_copy': StandardScaler(),
                               'acc_quarter_copy': StandardScaler(),
                               'num_payments': StandardScaler(),
                               'mean_payments': StandardScaler(),
                               'vco_payments': StandardScaler(),
                               'max_payment': StandardScaler(),
                               'num_revisions': StandardScaler(),
                               'mean_revisions': StandardScaler(),
                               'max_revision': StandardScaler(),
                               'total_revisions': StandardScaler(),
                               'prop_upward_revisions': StandardScaler(),
                               'incurred_copy': StandardScaler()}

                for key in self.scaler.keys():
                    if key in self.set.columns:
                        self.scaler[key].fit(self.set[key].values.reshape(-1, 1))
                        self.set[key] = self.scaler[key].transform(self.set[key].values.reshape(-1, 1))
                    
                    elif key in self.index.columns:
                        self.scaler[key].fit(self.index[key].values.reshape(-1, 1))
                        self.index[key] = self.scaler[key].transform(self.index[key].values.reshape(-1, 1))

                    else:
                        warnings.warn(f'{key} not found in either set or index dataframes')

            else:
                # apply scaling
                for key in self.scaler.keys():
                    if key in self.set.columns:
                        self.set[key] = self.scaler[key].transform(self.set[key].values.reshape(-1, 1))
                    
                    elif key in self.index.columns:
                        self.index[key] = self.scaler[key].transform(self.index[key].values.reshape(-1, 1))

                    else:
                        warnings.warn(f'{key} not found in either set or index dataframes')

        # removing latest 4/8 accident quarters from datasets to see what plots would look like without them
        # this is NOT intended to be a permanent change, just for visualisation purposes
        # this code appears at the end of the initialisation so that it does not interfere with the scaling above
        # self.index = self.index[self.index['acc_quarter'] <= 36]
        # self.set = self.set[self.set['index'].isin(self.index['index'])]
        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        # Retrieves time series data, as well as summary info
        # index runs from [0, __len__(self)]
        # real_index instead refers to indexes in the csv file

        real_index = self.index['index'][index]

        df = self.set[(self.set['index']==real_index)]

        print(f'Getting item {index}, real index {real_index}, df shape {df.shape}')
        if df.shape[0] == 0:
            raise ValueError(f'No data found for index {index} real index {real_index}')

        claim_no = df['claim_no'].mean()
        
        # Get relevant info from index.csv file
        target = self.index['target'][index]
        claim_size = self.index['claim_size'][index]
        latest_incurred = self.index['latest_incurred'][index]
        true_ocl = self.index['true_ocl'][index]
        dev_quarter = self.index['dev_quarter'][index]
        acc_quarter_copy = self.index['acc_quarter_copy'][index]
        pred_time_copy = self.index['pred_time_copy'][index]

        if self.include_covariates:
            legal_rep = self.index['Legal Representation'][index]
            injury_severity = self.index['Injury Severity'][index]
            claimant_age = self.index['Age of Claimant'][index]

        # Setting up the data to be input into an RNN model
        if self.model_type == 'RNN':
            nrows = df[(df['dev_time']!=0)\
                    | (df['cal_time']!=0)\
                    | (df['paid']!=0)\
                    | (df['ocl']!=0)].shape[0]
                
            if self.include_incurreds:
                databox = df[['dev_time', 'cal_time', 'paid', 'ocl']].copy()
                    
            else:
                databox = df[['dev_time','cal_time','paid']].copy()

            databox = torch.tensor(databox.values)

            # Return padded data
            if self.include_covariates:
                return (F.pad(databox.float(), (0,0,0,70-nrows)), # 70 is arbitrarily hard-coded, increase if you run into errors with mismatching dimensions
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, pred_time_copy, acc_quarter_copy, nrows, legal_rep, injury_severity, claimant_age)

            else:
                return (F.pad(databox.float(), (0,0,0,70-nrows)), 
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, pred_time_copy, acc_quarter_copy, nrows)

        # Setting up the data to be input into an FNN model
        elif self.model_type == 'FNN':
            num_payments = self.index['num_payments'][index]
            mean_payments = self.index['mean_payments'][index]
            vco_payments = self.index['vco_payments'][index]
            max_payment = self.index['max_payment'][index]
            
            # finding case estimate summary info
            if self.include_incurreds:
                num_revisions = self.index['num_revisions'][index]
                mean_revisions = self.index['mean_revisions'][index]
                max_revision = self.index['max_revision'][index]
                total_revisions = self.index['total_revisions'][index]
                prop_upward_revisions = self.index['prop_upward_revisions'][index]
                incurred_copy = self.index['incurred_copy'][index]

            if self.include_incurreds and self.include_covariates:

                return (pred_time_copy, dev_quarter, acc_quarter_copy, num_payments, mean_payments, vco_payments, max_payment, 
                        num_revisions, mean_revisions, max_revision, total_revisions, prop_upward_revisions, 
                        legal_rep, injury_severity, claimant_age,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, incurred_copy)
            
            elif self.include_incurreds and not self.include_covariates:

                return (pred_time_copy, dev_quarter, acc_quarter_copy, num_payments, mean_payments, vco_payments, max_payment, 
                        num_revisions, mean_revisions, max_revision, total_revisions, prop_upward_revisions,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, incurred_copy)
            
            elif not self.include_incurreds and self.include_covariates:

                return (pred_time_copy, dev_quarter, acc_quarter_copy, num_payments, mean_payments, vco_payments, max_payment,
                        legal_rep, injury_severity, claimant_age,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no)

            else:

                return (pred_time_copy, dev_quarter, acc_quarter_copy, num_payments, mean_payments, vco_payments, max_payment,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no)

        else:
            raise ValueError("model_type must be 'RNN' or 'FNN'")      

class ClaimsRNN(nn.Module):
    """
    Can use vanilla RNN, LSTM and GRU
    Can change this to experiement with different architectures/hyperparameters
    """

    def __init__(self, nHidden, nLayers, nOut, type='RNN', 
                 nonlinearity='relu', output_layer='linear', dropout=0.0, 
                 normalisation=True, include_incurreds=True, include_covariates=False):
        
        super(ClaimsRNN, self).__init__()
        self.nHidden = nHidden # number of hidden units
        self.nLayers = nLayers # number of layers
        self.type = type # either 'RNN', 'LSTM', or 'GRU'
        self.nonlinearity = nonlinearity # either 'relu' or 'tanh'
        # nonLinearity only used in vanilla RNN
        self.output_layer = output_layer # either 'linear' or 'exponential'
        self.dropout = dropout # float between 0 and 1
        self.include_incurreds = include_incurreds # needs to match ClaimsDataset
        self.relu = nn.ReLU() # used for feed-forward hidden layer, should change this so different activation functions can be specified
        self.include_covariates = include_covariates
        self.normalisation = normalisation # boolean for whether to use batch and layer normalisation
        #self.nConcatUnits = self.nHidden // 4 # this will be the number of units of both RNN and static inputs before concatenating


        # nFeatures is the number of features to be input into the RNN layer
        self.nFeatures = 3 + self.include_incurreds # 4 features with ocl, 3 without

        if self.include_covariates:
            self.embedding_dim = 2
        else:
            self.embedding_dim = 0
        
        self.nstatic = 2 + self.include_covariates * (1 + 2 * self.embedding_dim) # 2 guaranteed inputs (pred_time, acc_quarter) + legal rep + 2 covariate embeddings

        self.dropout_layer = nn.Dropout(self.dropout)

        if self.normalisation:
            #self.layer_norm1 = nn.LayerNorm(self.nFeatures)
            
            self.rnn_layers = nn.ModuleList()
            self.layer_norms_rnn = nn.ModuleList()

            for i in range(nLayers):
                input_size = self.nFeatures if i == 0 else self.nHidden

                if type == 'RNN':
                    self.rnn_layers.append(nn.RNN(input_size, nHidden, 1, 
                                                batch_first=True, nonlinearity=nonlinearity))
                elif type == 'LSTM':
                    self.rnn_layers.append(nn.LSTM(input_size, nHidden, 1, 
                                                batch_first=True))
                elif type == 'GRU':
                    self.rnn_layers.append(nn.GRU(input_size, nHidden, 1, 
                                                batch_first=True))
                else:
                    raise ValueError("type must be 'RNN', 'LSTM' or 'GRU'")

                self.layer_norms_rnn.append(nn.LayerNorm(nHidden))

            self.layer_norm2 = nn.LayerNorm(nHidden)

            #self.batch_norm1 = nn.BatchNorm1d(self.nConcatUnits)
            self.batch_norm2 = nn.BatchNorm1d(self.nstatic)
            self.batch_norm3 = nn.BatchNorm1d(self.nHidden // 2)

        else:
            if type == 'RNN':
                self.rnn = nn.RNN(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, nonlinearity=nonlinearity, 
                                dropout=dropout)

            elif type == 'LSTM':
                self.rnn = nn.LSTM(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, dropout=dropout) # device = device? should I add this?
            
            elif type == 'GRU':
                self.rnn = nn.GRU(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, dropout=dropout)
            
            else:
                raise ValueError("type must be 'RNN', 'LSTM' or 'GRU'")

        # RNN output is reduced in size
        #self.fc1 = nn.Linear(nHidden, self.nConcatUnits)

        if self.include_covariates:
            self.embedding_sev = nn.Embedding(6, self.embedding_dim) # 6 possible injury severities, output 2 dimensions
            self.embedding_age = nn.Embedding(5, self.embedding_dim) # 5 possible ages, output 2 dimensions
        
        self.fc3 = nn.Linear(self.nHidden + self.nstatic, self.nHidden // 2)

        self.fc4 = nn.Linear(self.nHidden // 2, nOut)

    def forward(self, x):
        # x[0] will be the packed datapoints, x[1:] will be the static covariates
        if self.normalisation:
            out = x[0]

            for i, rnn in enumerate(self.rnn_layers):
                out, ht = rnn(out)  # RNN output

                if i < self.nLayers - 1:
                    #print(f'Layer {i} output shape: {out.batch_sizes.size(0)}')
                    out, nrows = pad_packed_sequence(out, batch_first=True)
                    #print(f'Layer {i} padded output shape: {out.shape}')
                    out = self.layer_norms_rnn[i](out)
                    out = self.dropout_layer(out)
                    out = pack_padded_sequence(out, nrows, batch_first=True, enforce_sorted=False)
        
        else:
            out, ht = self.rnn(x[0])

        if self.type == 'LSTM':
            ht = ht[0]

        out = ht[-1,:,:]

        if self.normalisation:
            out = self.layer_norm2(out)

        if self.include_covariates:
            sev_embed = self.embedding_sev(x[4].long())
            age_embed = self.embedding_age(x[5].long())

            static_out = torch.cat((x[1], x[2], x[3], sev_embed[:, -1, :], age_embed[:, -1, :]), 1)
        
        else:
            static_out = torch.cat((x[1], x[2]), 1)

        if self.normalisation:
            static_out = self.batch_norm2(static_out)

        out = torch.cat((out, static_out), 1)

        out = self.fc3(out)

        if self.normalisation:
            out = self.batch_norm3(out)    

        out = self.relu(out)

        out = self.dropout_layer(out)          

        out = self.fc4(out)

        if self.output_layer == 'exponential':
            out = torch.exp(out)

        elif self.output_layer == 'softplus':
            out = F.softplus(out)

        return out
    

class ClaimsFNN(nn.Module):
    """
    Feed-Forward Neural Network to be used as a benchmark model
    """

    def __init__(self, nLayers=2, nHidden=50, dropout = 0.2, 
                 final_activation='exponential', normalisation=True, 
                 include_incurreds=True, include_covariates=True):
        """
        Initialises the Feedforward Neural Network.

        Args:
            p: Number of features in the input dataset.
            nLayers: Number of hidden layers in the network.
            nHidden: Number of neurons in each hidden layer.
            dropout: Dropout rate for regularization.
            final_activation: Output layer activation function. 'exp' for exponential, 'linear' for linear.
            normalisation: Whether to normalise inputs or not
        """
        super(ClaimsFNN, self).__init__()

        self.include_covariates = include_covariates

        if self.include_covariates:
            self.embedding_dim = 2
            self.embedding_sev = nn.Embedding(6, self.embedding_dim) # 6 possible injury severities, output 2 dimensions
            self.embedding_age = nn.Embedding(5, self.embedding_dim) # 5 possible ages, output 2 dimensions

        # 3 variables for transaction times, 4 for payments, 6 for revisions, 1 raw covariate, 2 covariate embeddings
        self.num_features = 7 + 6 * include_incurreds + include_covariates * (1 + 2 * self.embedding_dim)
        self.final_activation = final_activation
        self.normalisation = normalisation

        if self.normalisation:
            layers = [nn.Linear(self.num_features, nHidden), nn.BatchNorm1d(nHidden), nn.ReLU(), nn.Dropout(dropout)]

            for _ in range(nLayers - 1):
                layers.append(nn.Linear(nHidden, nHidden))
                layers.append(nn.BatchNorm1d(nHidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        else:
            layers = [nn.Linear(self.num_features, nHidden), nn.ReLU(), nn.Dropout(dropout)]

            for _ in range(nLayers - 1):
                layers.append(nn.Linear(nHidden, nHidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))  # Add dropout after each activation

        layers.append(nn.Linear(nHidden, 1))
        self.nn_output_layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted outputs for the distributions.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted outputs (shape: (n,))
        """
        if self.include_covariates:
            sev_embed = self.embedding_sev(x[:, -2].long())
            age_embed = self.embedding_age(x[:, -1].long())

            out = torch.cat((x[:, :-2], sev_embed, age_embed), 1)

        else:
            out = x

        if self.final_activation == 'exponential':
            out = torch.exp(self.nn_output_layer(out).squeeze(-1)) # * x[:, -1].squeeze(-1)
        elif self.final_activation == 'softplus':
            out = F.softplus(self.nn_output_layer(out).squeeze(-1))
        elif self.final_activation == 'linear':
            out = self.nn_output_layer(out).squeeze(-1)
        else:
            raise ValueError(f"Unsupported final activation function: {self.final_activation}")
        assert out.shape == torch.Size([x.shape[0]])
        return out

### TRAINING/TESTING FUNCTIONS ################################################

def train_network(model, train_data, hp_comb, verbose=True, 
                  val_data=None, cv_loss_list=None, cv_vsInc_list=None, 
                  cv_weighted_vsInc_claimsize_list=None, 
                  cv_weighted_vsInc_ocl_list=None, cv_uie_list=None):
    
    """
    Args:
        model: the model to train
        train_data: ClaimsDataset object containing training data
        hp_comb: dictionary of hyperparameters along with their values
        verbose: whether to print written outputs and progress to console
        val_data: ClaimsDataset object containing validation data, 
            enables early stopping
        cv_loss_list/cv_vsInc_list/cv_uie_list: lists to be passed to keep 
            track of between-model stats during cross-validation
    
    """

    if val_data is not None:
        train_loss_list = []
        train_weighted_vsInc_ocl_list = []
        train_agg_clmsize_percent_error_model = []

        val_loss_list = []
        val_vsInc_list = []
        val_weighted_vsInc_claimsize_list = []
        val_weighted_vsInc_ocl_list = []
        val_uie_list = []
        val_agg_clmsize_percent_error_model = []

        best_val_loss = np.inf
        best_val_vsInc = 0
        best_val_weighted_vsInc_claimsize = 0
        best_val_weighted_vsInc_ocl = 0
        best_val_uie = np.inf
        best_weights = None

        patience_counter = 0

    # Data loader
    trainloader = torch.utils.data.DataLoader(dataset=train_data, 
                                              batch_size=hp_comb['batch_size'], 
                                              shuffle=True, drop_last=False,
                                              num_workers=4, pin_memory=True)
    
    # Creation of optimiser
    if hp_comb['optimiser'] == 'Adam':
        optimiser = optim.Adam(model.parameters(), lr=hp_comb['lr'])

    elif hp_comb['optimiser'] == 'AdamW':
        optimiser = optim.AdamW(model.parameters(), lr=hp_comb['lr'])

    else:
        raise ValueError("optimiser must be 'Adam' or 'AdamW'. Otherwise, add new optimiser to the function.")

    # Train the model
    for epoch in range(hp_comb['epochs']):
        #print('new epoch')

        total_loss = 0
        total_datapoints = 0
        total_vsInc = 0
        total_weighted_vsInc_claimsize = 0
        total_weighted_vsInc_ocl = 0
        total_uie = 0
        total_ultimates = 0
        total_ocls = 0
        total_preds = 0
        total_incurreds = 0

        for batch in trainloader:

            #print('new batch')

            if hp_comb['model_type'] == 'RNN':
                # extract batch data
                if hp_comb['include_covariates']:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls,
                    indexes, claim_nos, pred_time_copys, acc_quarter_copys, nrowss, legal_reps, 
                    injury_severities, claimant_ages) = batch

                    legal_reps = legal_reps.unsqueeze(1).to(device).float()
                    injury_severities = injury_severities.unsqueeze(1).to(device).float()
                    claimant_ages = claimant_ages.unsqueeze(1).to(device).float()
                    
                else:
                    (datapoints, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos, pred_time_copys, acc_quarter_copys, nrowss) = batch
                    
                datapoints = datapoints.to(device).float()
                targets = targets.to(device).float()
                claim_sizes = claim_sizes.to(device).float()
                latest_incurreds = latest_incurreds.to(device).float()
                true_ocls = true_ocls.to(device).float()
                pred_time_copys = pred_time_copys.unsqueeze(1).to(device).float()
                acc_quarter_copys = acc_quarter_copys.unsqueeze(1).to(device).float()

                packed = pack_padded_sequence(datapoints, nrowss, 
                                            enforce_sorted=False, 
                                            batch_first=True)

                if hp_comb['include_covariates']:
                    packed_extra = (packed, pred_time_copys, acc_quarter_copys, legal_reps, 
                                    injury_severities, claimant_ages)

                else:
                    packed_extra = (packed, pred_time_copys, acc_quarter_copys)  

            elif hp_comb['model_type'] == 'FNN':
                if hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, num_revisionss, mean_revisionss, max_revisions, 
                    total_revisionss, prop_upward_revisionss, legal_reps, 
                    injury_severities, claimant_ages, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos, incurred_copys) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    num_revisionss = num_revisionss.to(device).float()
                    mean_revisionss = mean_revisionss.to(device).float()
                    max_revisions = max_revisions.to(device).float()
                    total_revisionss = total_revisionss.to(device).float()
                    prop_upward_revisionss = prop_upward_revisionss.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()
                    incurred_copys = incurred_copys.to(device).float()

                    #print(f'type of pred_times: {type(pred_times)}')
                    #print(f'dimension of pred_times: {pred_times.shape}')

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, incurred_copys, num_revisionss, mean_revisionss, max_revisions,
                                    total_revisionss, prop_upward_revisionss, legal_reps,
                                    injury_severities, claimant_ages), dim=1).to(device)

                    #print(f'dimension of packed_extra: {packed_extra.shape}')

                elif hp_comb['include_incurreds'] and not hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, num_revisionss, mean_revisionss, max_revisions, 
                    total_revisionss, prop_upward_revisionss, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos, incurred_copys) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    num_revisionss = num_revisionss.to(device).float()
                    mean_revisionss = mean_revisionss.to(device).float()
                    max_revisions = max_revisions.to(device).float()
                    total_revisionss = total_revisionss.to(device).float()
                    prop_upward_revisionss = prop_upward_revisionss.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()
                    incurred_copys = incurred_copys.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, incurred_copys, num_revisionss, mean_revisionss, max_revisions,
                                    total_revisionss, prop_upward_revisionss), dim=1).to(device)

                elif not hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, legal_reps, injury_severities, 
                    claimant_ages, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, legal_reps, injury_severities, 
                                    claimant_ages), dim=1).to(device)

                else:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, 
                                                num_paymentss, mean_paymentss,
                                                vco_paymentss, max_payments), dim=1).to(device)

            else:
                raise ValueError("model_type must be 'RNN' or 'FNN'")

            raw_preds = model(packed_extra)    
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            # undoing any scaling
            if train_data.scaler is not None:
                target_mean = train_data.scaler['target'].mean_[0]
                target_std = train_data.scaler['target'].scale_[0]

                preds = raw_preds * target_std + target_mean
                ultimates = targets * target_std + target_mean

            else:
                preds = raw_preds
                ultimates = targets

            #print(f'mean raw preds: {raw_preds.mean()}')
            #print(f'mean raw targets: {targets.mean()}')

            # converting raw preds and targets to be in terms of ultimate claim size
            if train_data.target_col == 'claim_size':
                preds = preds
                ultimates = ultimates
            
            elif train_data.target_col == 'log_claim_size':
                preds = torch.exp(preds)
                ultimates = torch.exp(ultimates)
            
            elif train_data.target_col == 'log_m':
                preds = torch.exp(preds) * latest_incurreds
                ultimates = torch.exp(ultimates) * latest_incurreds
            
            elif train_data.target_col == 'true_ocl':
                preds = preds + claim_sizes - true_ocls
                ultimates = ultimates + claim_sizes - true_ocls

            elif train_data.target_col == 'log_true_ocl':
                preds = torch.exp(preds) + claim_sizes - true_ocls
                ultimates = torch.exp(ultimates) + claim_sizes - true_ocls

            else:
                ValueError('Invalid target, must be "claim_size", "log_claim_size", "log_m", "true_ocl" or "log_true_ocl"')

            #print(f'mean raw preds: {raw_preds.mean()}, mean raw targets: {targets.mean()}')
            #print(f'mean preds: {preds.mean()}, mean ultimates: {ultimates.mean()}, mean incurreds: {latest_incurreds.mean()}')

            # Loss and gradient descent
            if isinstance(hp_comb['criterion'], MSLE_with_penalty):
                loss = hp_comb['criterion'](raw_preds, targets, claim_sizes - true_ocls, preds)

            else:
                loss = hp_comb['criterion'](raw_preds, targets)
            
            optimiser.zero_grad()
            loss.backward() # Calculate gradients

            # clip gradients
            max_norm = 5
            clip_grad_norm_(model.parameters(), max_norm)

            optimiser.step() # Update weights

            # Track statistics
            total_loss += loss.detach() * preds.size(0)
            total_datapoints += preds.size(0)
            total_vsInc += sum(torch.abs((ultimates-preds)) < 
                               torch.abs((ultimates-latest_incurreds)))
            
            total_weighted_vsInc_claimsize += sum(ultimates * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_weighted_vsInc_ocl += sum(true_ocls * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_preds += sum(preds)
            total_incurreds += sum(latest_incurreds)
            total_ultimates += sum(ultimates)
            total_ocls += sum(true_ocls)
            
            # uie is not being included in the paper, but useful as a diagnostic during training
            total_uie+=sum(torch.logical_and((preds < latest_incurreds), 
                                             (torch.abs((ultimates-preds)) > 
                                              torch.abs((ultimates-
                                                         latest_incurreds)))))

        # End of epoch summary
        vs_incurred_accuracy = (total_vsInc / total_datapoints * 100).item()
        uie = (total_uie / total_datapoints * 100).item()
        total_loss = (total_loss / total_datapoints).item()
        weighted_vsinc_claimsize = (total_weighted_vsInc_claimsize / total_ultimates * 100).item()
        weighted_vsinc_ocl = (total_weighted_vsInc_ocl / total_ocls * 100).item()
        agg_clmsize_percent_error_model = ((total_preds - total_incurreds) / total_ultimates * 100).item()
        agg_clmsize_percent_error_incurreds = ((total_incurreds - total_ultimates) / total_ultimates * 100).item()

        if verbose:
            print(f'Epoch {epoch}: '
                  f'training loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc (Claim Size) = {weighted_vsinc_claimsize:.2f}%, '
                  f'weighted vsInc (OCL) = {weighted_vsinc_ocl:.2f}%, '
                  f'UIE = {uie:.2f}%, '
                  f'model aggregate error = {agg_clmsize_percent_error_model:.2f}%, '
                  f'incurreds aggregate error = {agg_clmsize_percent_error_incurreds:.2f}%')
            
        if isinstance(train_loss_list, list):
            train_loss_list.append(total_loss)

        if isinstance(train_weighted_vsInc_ocl_list, list):
            train_weighted_vsInc_ocl_list.append(weighted_vsinc_ocl)

        if isinstance(train_agg_clmsize_percent_error_model, list):
            train_agg_clmsize_percent_error_model.append(agg_clmsize_percent_error_model)

        # Validation
        if val_data:
            
            if verbose:
                print('Validation')
                
            test_network(model, val_data, hp_comb, val_loss_list=val_loss_list, 
                         val_vsInc_list=val_vsInc_list, 
                         val_weighted_vsInc_claimsize_list=val_weighted_vsInc_claimsize_list,
                         val_weighted_vsInc_ocl_list=val_weighted_vsInc_ocl_list,
                         val_uie_list=val_uie_list,
                         val_agg_clmsize_percent_error_model=val_agg_clmsize_percent_error_model, 
                         verbose=verbose)

            # Early stopping
            min_delta = 0.0001

            if val_loss_list[-1] < best_val_loss - min_delta:
                best_val_loss = val_loss_list[-1]
                best_val_vsInc = val_vsInc_list[-1]
                best_val_weighted_vsInc_claimsize = val_weighted_vsInc_claimsize_list[-1]
                best_val_weighted_vsInc_ocl = val_weighted_vsInc_ocl_list[-1]
                best_val_uie = val_uie_list[-1]
                patience_counter = 0
                best_weights = deepcopy(model.state_dict())

            else:
                patience_counter += 1

            if patience_counter == hp_comb['patience']:
                model.load_state_dict(best_weights)
                if cv_loss_list is not None:
                    cv_loss_list.append(best_val_loss)
                if cv_vsInc_list is not None:
                    cv_vsInc_list.append(best_val_vsInc)
                if cv_weighted_vsInc_claimsize_list is not None:
                    cv_weighted_vsInc_claimsize_list.append(best_val_weighted_vsInc_claimsize)
                if cv_weighted_vsInc_ocl_list is not None:
                    cv_weighted_vsInc_ocl_list.append(best_val_weighted_vsInc_ocl)
                if cv_uie_list is not None:
                    cv_uie_list.append(best_val_uie)
                
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch}')
                    print(f'Validation: '
                          f'loss = {round_threshold(best_val_loss):,}, '
                          f'vsInc = {best_val_vsInc:.2f}%, '
                          f'weighted vsInc (Claim Size) = {best_val_weighted_vsInc_claimsize:.2f}%, '
                          f'weighted vsInc (OCL) = {best_val_weighted_vsInc_ocl:.2f}%, '
                          f'UIE = {best_val_uie:.2f}%\n')
                    
                    # produce epoch graphs
                    '''plt.plot(list(range(epoch + 1)), train_loss_list, label='train loss', color='blue')
                    plt.xlabel('epoch')
                    plt.ylabel('training loss')
                    plt.title('Training loss curve over epochs')
                    plt.show()

                    plt.plot(list(range(epoch + 1)), val_loss_list, label='val loss', color='orange')
                    plt.xlabel('epoch')
                    plt.ylabel('validation loss')
                    plt.title('Validation loss curve over epochs')
                    plt.show()

                    # plotting vsInc curves together because they are hopefully on similar scales
                    plt.plot(list(range(epoch + 1)), train_weighted_vsInc_ocl_list, label='train vsInc (OCL)', color='blue')
                    plt.plot(list(range(epoch + 1)), val_weighted_vsInc_ocl_list, label='val vsInc (OCL)', color='orange')
                    plt.xlabel('epoch')
                    plt.ylabel('vsInc (OCL)')
                    plt.title('vsInc (OCL) curve over epochs')
                    plt.legend()
                    plt.show()

                    # ignoring the first 5 epochs for aggregate error curves because they are too noisy
                    plt.plot(list(range(5, epoch + 1)), train_agg_clmsize_percent_error_model[5:], label='train agg error', color='blue')
                    plt.plot(list(range(5, epoch + 1)), val_agg_clmsize_percent_error_model[5:], label='val agg error', color='orange')
                    plt.xlabel('epoch')
                    plt.ylabel('aggregate error (%)')
                    plt.title('Aggregate error curve over epochs')
                    plt.legend()
                    plt.show()'''
                    
                break

        # if we reach max number of epochs, save the best weights
        if ((epoch == hp_comb['epochs'] - 1) and 
            (val_data is not None) and 
            (cv_loss_list is not None) and 
            (cv_vsInc_list is not None) and
            (cv_weighted_vsInc_claimsize_list is not None) and
            (cv_weighted_vsInc_ocl_list is not None) and
            (cv_uie_list is not None)):

            model.load_state_dict(best_weights)
            cv_loss_list.append(best_val_loss)
            cv_vsInc_list.append(best_val_vsInc)
            cv_weighted_vsInc_claimsize_list.append(best_val_weighted_vsInc_claimsize)
            cv_weighted_vsInc_ocl_list.append(best_val_weighted_vsInc_ocl)
            cv_uie_list.append(best_val_uie)

            if verbose:
                print(f'\nNo early stopping')
                print(f'Validation: loss = {round_threshold(best_val_loss):,}, '
                    f'vsInc = {best_val_vsInc:.2f}%, '
                    f'weighted vsInc (Claim Size) = {best_val_weighted_vsInc_claimsize:.2f}%, '
                    f'weighted vsInc (OCL) = {best_val_weighted_vsInc_ocl:.2f}%, '
                    f'UIE = {best_val_uie:.2f}%\n')
                
                # produce epoch graphs
                '''plt.plot(list(range(epoch + 1)), train_loss_list, label='train loss', color='blue')
                plt.xlabel('epoch')
                plt.ylabel('training loss')
                plt.title('Training loss curve over epochs')
                plt.show()

                plt.plot(list(range(epoch + 1)), val_loss_list, label='val loss', color='orange')
                plt.xlabel('epoch')
                plt.ylabel('validation loss')
                plt.title('Validation loss curve over epochs')
                plt.show()

                # plotting vsInc curves together because they are hopefully on similar scales
                plt.plot(list(range(epoch + 1)), train_weighted_vsInc_ocl_list, label='train vsInc (OCL)', color='blue')
                plt.plot(list(range(epoch + 1)), val_weighted_vsInc_ocl_list, label='val vsInc (OCL)', color='orange')
                plt.xlabel('epoch')
                plt.ylabel('vsInc (OCL)')
                plt.title('vsInc (OCL) curve over epochs')
                plt.legend()
                plt.show()

                # ignoring the first 5 epochs for aggregate error curves because they are too noisy
                plt.plot(list(range(5, epoch + 1)), train_agg_clmsize_percent_error_model[5:], label='train agg error', color='blue')
                plt.plot(list(range(5, epoch + 1)), val_agg_clmsize_percent_error_model[5:], label='val agg error', color='orange')
                plt.xlabel('epoch')
                plt.ylabel('aggregate error (%)')
                plt.title('Aggregate error curve over epochs')
                plt.legend()
                plt.show()'''
            
def test_network(model, test_data, hp_comb, preds_list=None, verbose=True, 
                 val_loss_list=None, val_vsInc_list=None, 
                 val_weighted_vsInc_claimsize_list=None, 
                 val_weighted_vsInc_ocl_list=None, val_uie_list=None,
                 val_agg_clmsize_percent_error_model=None):
    
    """Args:
        preds_list: empty list to append predictions to
        val_loss_list/val_vsInc_list/val_uie_list: lists to be passed to keep 
        track of within-model stats during training
    """

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                              batch_size=hp_comb['batch_size'], 
                                              shuffle=False, drop_last=False,
                                              num_workers=4, pin_memory=True)

    total_loss = 0
    total_datapoints = 0
    total_vsInc = 0
    total_weighted_vsInc_claimsize = 0
    total_weighted_vsInc_ocl = 0
    total_uie = 0
    total_ultimates = 0
    total_ocls = 0
    total_preds = 0
    total_incurreds = 0

    # set model to test mode
    model.eval()

    # Test the model
    with torch.no_grad():
        for batch in test_loader:

            if hp_comb['model_type'] == 'RNN':

                if hp_comb['include_covariates']:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                    indexes, claim_nos, pred_time_copys, acc_quarter_copys, nrowss, legal_reps, 
                    injury_severities, claimant_ages) = batch

                    legal_reps = legal_reps.unsqueeze(1).to(device).float()
                    injury_severities = injury_severities.unsqueeze(1).to(device).float()
                    claimant_ages = claimant_ages.unsqueeze(1).to(device).float()

                else:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                    indexes, claim_nos, pred_time_copys, acc_quarter_copys, nrowss) = batch

                datapoints = datapoints.to(device).float()
                targets = targets.to(device).float()
                claim_sizes = claim_sizes.to(device).float()
                latest_incurreds = latest_incurreds.to(device).float()
                true_ocls = true_ocls.to(device).float()
                pred_time_copys = pred_time_copys.unsqueeze(1).to(device).float()
                acc_quarter_copys = acc_quarter_copys.unsqueeze(1).to(device).float()

                #print(datapoints)
                #print(nrowss)

                packed = pack_padded_sequence(datapoints, nrowss, 
                                            enforce_sorted=False, 
                                            batch_first=True)

                if hp_comb['include_covariates']:
                    packed_extra = (packed, pred_time_copys, acc_quarter_copys, legal_reps, 
                                    injury_severities, claimant_ages)
                    
                else:
                    packed_extra = (packed, pred_time_copys, acc_quarter_copys)

            elif hp_comb['model_type'] == 'FNN':
                if hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, num_revisionss, mean_revisionss, max_revisions, 
                    total_revisionss, prop_upward_revisionss, legal_reps, 
                    injury_severities, claimant_ages, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos, incurred_copys) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    num_revisionss = num_revisionss.to(device).float()
                    mean_revisionss = mean_revisionss.to(device).float()
                    max_revisions = max_revisions.to(device).float()
                    total_revisionss = total_revisionss.to(device).float()
                    prop_upward_revisionss = prop_upward_revisionss.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()
                    incurred_copys = incurred_copys.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, incurred_copys, num_revisionss, mean_revisionss, max_revisions,
                                    total_revisionss, prop_upward_revisionss, legal_reps,
                                    injury_severities, claimant_ages), dim=1).to(device)

                elif hp_comb['include_incurreds'] and not hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, num_revisionss, mean_revisionss, max_revisions, 
                    total_revisionss, prop_upward_revisionss, targets, claim_sizes,
                    latest_incurreds, true_ocls, indexes, claim_nos, incurred_copys) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    num_revisionss = num_revisionss.to(device).float()
                    mean_revisionss = mean_revisionss.to(device).float()
                    max_revisions = max_revisions.to(device).float()
                    total_revisionss = total_revisionss.to(device).float()
                    prop_upward_revisionss = prop_upward_revisionss.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()
                    incurred_copys = incurred_copys.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, incurred_copys, num_revisionss, mean_revisionss, max_revisions,
                                    total_revisionss, prop_upward_revisionss), dim=1).to(device)

                elif not hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, legal_reps, injury_severities, 
                    claimant_ages, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss,
                                    vco_paymentss, max_payments, legal_reps, injury_severities, 
                                    claimant_ages), dim=1).to(device)
                
                else:
                    (pred_time_copys, dev_quarters, acc_quarter_copys, num_paymentss, mean_paymentss, 
                    vco_paymentss, max_payments, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_time_copys = pred_time_copys.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarter_copys = acc_quarter_copys.to(device).float()
                    num_paymentss = num_paymentss.to(device).float()
                    mean_paymentss = mean_paymentss.to(device).float()
                    vco_paymentss = vco_paymentss.to(device).float()
                    max_payments = max_payments.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_time_copys, dev_quarters, acc_quarter_copys, 
                                                num_paymentss, mean_paymentss,
                                                vco_paymentss, max_payments), dim=1).to(device)

            else:
                raise ValueError("model_type must be 'RNN' or 'FNN'")

            raw_preds = model(packed_extra)
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            # undoing any scaling
            if test_data.scaler is not None:
                target_mean = test_data.scaler['target'].mean_[0]
                target_std = test_data.scaler['target'].scale_[0]

                preds = raw_preds * target_std + target_mean
                ultimates = targets * target_std + target_mean
            
            else:
                preds = raw_preds
                ultimates = targets

            # converting raw preds and targets to be in terms of ultimate claim size
            if test_data.target_col == 'claim_size':
                preds = preds
                ultimates = ultimates

            elif test_data.target_col == 'log_claim_size':
                preds = torch.exp(preds)
                ultimates = torch.exp(ultimates)
            
            elif test_data.target_col == 'log_m':
                preds = torch.exp(preds) * latest_incurreds
                ultimates = torch.exp(ultimates) * latest_incurreds

            elif test_data.target_col == 'true_ocl':
                preds = preds + claim_sizes - true_ocls
                ultimates = ultimates + claim_sizes - true_ocls

            elif test_data.target_col == 'log_true_ocl':
                preds = torch.exp(preds) + claim_sizes - true_ocls
                ultimates = torch.exp(ultimates) + claim_sizes - true_ocls

            else:
                ValueError('Invalid target, must be "claim_size", "log_claim_size", "log_m", "true_ocl" or "log_true_ocl"')

            #print(f'mean raw preds: {raw_preds.mean()}, mean raw targets: {targets.mean()}')
            #print(f'mean preds: {preds.mean()}, mean ultimates: {ultimates.mean()}, mean incurreds: {latest_incurreds.mean()}')

            # Loss and gradient descent
            if isinstance(hp_comb['criterion'], MSLE_with_penalty):
                loss = hp_comb['criterion'](raw_preds, targets, claim_sizes - true_ocls, preds)

            else:
                loss = hp_comb['criterion'](raw_preds, targets)

            # Track statistics
            total_loss += loss.detach() * preds.size(0)
            total_datapoints += preds.size(0)
            total_vsInc += sum(torch.abs((ultimates-preds)) < 
                               torch.abs((ultimates-latest_incurreds)))
            
            total_weighted_vsInc_claimsize += sum(ultimates * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_weighted_vsInc_ocl += sum(true_ocls * (torch.abs((ultimates-preds)) <
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_ultimates += sum(ultimates)
            total_ocls += sum(true_ocls)
            
            total_uie+=sum(torch.logical_and((preds < latest_incurreds), 
                                             (torch.abs((ultimates-preds)) > 
                                              torch.abs((ultimates-
                                                         latest_incurreds)))))
            
            total_preds += sum(preds)
            total_incurreds += sum(latest_incurreds)

            if preds_list is not None:
                preds_list.extend([pred.item() for pred in preds])

        # End of epoch summary
        vs_incurred_accuracy = (total_vsInc / total_datapoints * 100).item()
        uie = (total_uie / total_datapoints * 100).item()
        total_loss = (total_loss / total_datapoints).item()
        weighted_vsinc_claimsize = (total_weighted_vsInc_claimsize / total_ultimates * 100).item()
        weighted_vsinc_ocl = (total_weighted_vsInc_ocl / total_ocls * 100).item()
        agg_clmsize_percent_error_model = ((total_preds - total_incurreds) / total_ultimates * 100).item()
        agg_clmsize_percent_error_incurreds = ((total_incurreds - total_ultimates) / total_ultimates * 100).item()

        if verbose:
            print(f'loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc (Claim Size) = {weighted_vsinc_claimsize:.2f}%, '
                  f'weighted vsInc (OCL) = {weighted_vsinc_ocl:.2f}%, '
                  f'UIE = {uie:.2f}%, '
                  f'model aggregate error = {agg_clmsize_percent_error_model:.2f}%, '
                  f'incurreds aggregate error = {agg_clmsize_percent_error_incurreds:.2f}%')

        if isinstance(val_loss_list, list):
            val_loss_list.append(total_loss)

        if isinstance(val_vsInc_list, list):
            val_vsInc_list.append(vs_incurred_accuracy)

        if isinstance(val_weighted_vsInc_claimsize_list, list):
            val_weighted_vsInc_claimsize_list.append(weighted_vsinc_claimsize)

        if isinstance(val_weighted_vsInc_ocl_list, list):
            val_weighted_vsInc_ocl_list.append(weighted_vsinc_ocl)

        if isinstance(val_uie_list, list):
            val_uie_list.append(uie)

        if isinstance(val_agg_clmsize_percent_error_model, list):
            val_agg_clmsize_percent_error_model.append(agg_clmsize_percent_error_model)

    # set model back to training mode
    model.train()

def get_heatmap(actuals, preds, nbins):
    '''Creates a heatmap between the actuals and the predictions on a 
       log10 scale'''
    H, actuals_edges, preds_edges = np.histogram2d(np.log10(actuals), 
                                                   np.log10(preds), bins=nbins)

    plt.pcolormesh(actuals_edges, preds_edges, H.T)
    plt.axline([np.log10(actuals.median()), np.log10(actuals.median())], 
               slope=1, color='red')
    
    plt.xlabel('Actual ($log_{10}$)')
    plt.ylabel('Predicted ($log_{10}$)')
    plt.show()

def get_losses(actuals, preds, incurreds, dataset, hp_comb):
    '''Computes and prints the MALE and MSLE for the model's predictions and 
    the case estimates.
    
    The losses will be calculated using ultimate claim size for 'claim_size' 
    and 'log_m' targets, and true ocl for 'true_ocl' and 'log_true_ocl' targets.'''

    if hp_comb['target_col'] == 'true_ocl' or hp_comb['target_col'] == 'log_true_ocl':
        actuals = dataset.index['true_ocl']
        preds = preds - dataset.index['claim_size'] + dataset.index['true_ocl']
        incurreds = incurreds - dataset.index['claim_size'] + dataset.index['true_ocl']

    preds_male = MeanAbsoluteLogError()(preds, actuals)
    preds_msle = MeanSquaredLogError()(preds, actuals)

    incurreds_male = MeanAbsoluteLogError()(incurreds, actuals)
    incurreds_msle = MeanSquaredLogError()(incurreds, actuals)

    print(f'model MALE: {preds_male:.3f}, MSLE: {preds_msle:.3f}')
    print(f'incurred MALE: {incurreds_male:.3f}, MSLE: {incurreds_msle:.3f}')


def get_vsInc(actuals, preds, incurreds):
    return 100 * np.mean(np.abs((actuals-preds)) < np.abs((actuals-incurreds)))

def get_weighted_vsInc_claimsize(actuals, preds, incurreds):
    if sum(actuals) > 0:
        return 100 * np.average(np.abs((actuals-preds)) < 
                                np.abs((actuals-incurreds)), weights=actuals)
    else:
        return None

def get_weighted_vsInc_ocl(actuals, preds, incurreds, ocls):
    if sum(ocls) > 0:
        return 100 * np.average(np.abs((actuals-preds)) < 
                                np.abs((actuals-incurreds)), weights=ocls)
    
    else:
        return None

def get_preds_actuals(model, test_data, param_dict, verbose=False):
    '''Note: 'actuals' refers to the ultimate claim size
    
        Args:
         model: the trained model
         test_data: the test dataset
         param_dict: dictionary of parameters used in the model
         verbose: whether to print written outputs and progress to console
         
       Output:
         actuals_list: list of actual claim sizes
         preds_list: list of model's predictions
         incurreds_list: list of case estimates'''
    
    preds_list = []

    test_network(model, test_data, param_dict, preds_list=preds_list, verbose=verbose, 
                 val_loss_list=None, val_vsInc_list=None, 
                 val_weighted_vsInc_claimsize_list=None, 
                 val_weighted_vsInc_ocl_list=None, val_uie_list=None,
                 val_agg_clmsize_percent_error_model=None)

    preds_list = pd.Series(preds_list)
    actuals_list = test_data.index["claim_size"]
    incurreds_list = test_data.index["latest_incurred"]
    ocls_list = test_data.index["true_ocl"]

    return actuals_list, preds_list, incurreds_list, ocls_list

def get_latest(test_data, actuals, preds, incurreds, ocls):
    '''Finds the latest prediction for each claim and returns the latest model
       predictions, and associated actuals and case estimates'''

    claim_indices = {}
    for index, claim_no in enumerate(test_data.index['claim_no']):
        claim_indices[claim_no] = index

    indicator = np.zeros(len(preds), dtype=bool)
    for claim_no in test_data.index['claim_no'].unique():
        max_index = claim_indices[claim_no]
        indicator[max_index] = True

    latest_preds = np.array(preds)[indicator]
    latest_actuals = actuals[indicator]
    latest_incurreds = incurreds[indicator]
    latest_ocls = ocls[indicator]

    latest_data = deepcopy(test_data)
    latest_data.index = test_data.index[indicator]

    return latest_actuals, latest_preds, latest_incurreds, latest_ocls, latest_data

def get_dev_quarter(test_data, actuals, preds, incurreds, ocls, dev_quarter):
    '''Finds the predictions for a specific development quarter and returns the
       model predictions, and associated actuals and case estimates'''
    indicator = test_data.index['dev_quarter'] == dev_quarter
    dev_actuals = actuals[indicator]
    dev_preds = preds[indicator]
    dev_incurreds = incurreds[indicator]
    dev_ocls = ocls[indicator]

    dev_data = deepcopy(test_data)
    dev_data.index = test_data.index[indicator]

    return dev_actuals, dev_preds, dev_incurreds, dev_ocls, dev_data

def extract_performance_by_time(index_data, actuals, preds, incurreds, ocls, time_str):
# 1 dataset, 1 prediction
    if isinstance(preds, pd.Series):
        times = np.sort(index_data[time_str].unique())

        actuals_by_time = np.zeros(len(times))
        incurreds_by_time = np.zeros(len(times))
        ocls_by_time = np.zeros(len(times))
        paids_by_time = np.zeros(len(times))

        preds_by_time = np.zeros(len(times))
        vsInc_by_time = np.zeros(len(times))
        weighted_vsInc_claimsize_by_time = np.zeros(len(times))
        weighted_vsInc_ocl_by_time = np.zeros(len(times))

        # these will not be used for 1 dataset, 1 prediction but need to define them for the results dictionary
        ocl_preds_by_time = None
        ocl_incurreds_by_time = None
        preds_over_actuals_by_time = None
        incurreds_over_actuals_by_time = None
        ocl_preds_over_actuals_by_time = None
        ocl_preds_over_actuals_by_time_adjusted = None
        ocl_incurreds_over_actuals_by_time = None

    else:
        
        # 1 dataset, multiple predictions
        if isinstance(actuals, pd.Series):
            times = np.sort(index_data[time_str].unique())

            actuals_by_time = np.zeros(len(times))
            incurreds_by_time = np.zeros(len(times))
            ocls_by_time = np.zeros(len(times))
            paids_by_time = np.zeros(len(times))

            ocl_preds_by_time = np.zeros((len(preds), len(times)))
            ocl_incurreds_by_time = np.zeros(len(times))

            # these will not be used for 1 dataset, multiple predictions but need to define them for the results dictionary
            preds_over_actuals_by_time = None
            incurreds_over_actuals_by_time = None
            ocl_preds_over_actuals_by_time = None
            ocl_preds_over_actuals_by_time_adjusted = None
            ocl_incurreds_over_actuals_by_time = None

        # multiple datasets, multiple predictions
        else:
            times_list = []
            
            for i in range(len(preds)):
                times_list.append(index_data[i][time_str].unique())

            times = np.sort(np.unique(np.concatenate(times_list)))

            actuals_by_time = np.zeros((len(actuals), len(times)))
            incurreds_by_time = np.zeros((len(incurreds), len(times)))
            ocls_by_time = np.zeros((len(ocls), len(times)))
            paids_by_time = np.zeros((len(ocls), len(times)))

            preds_over_actuals_by_time = np.zeros((len(preds), len(times)))
            incurreds_over_actuals_by_time = np.zeros((len(incurreds), len(times)))

            ocl_preds_over_actuals_by_time = np.zeros((len(preds), len(times)))
            ocl_preds_over_actuals_by_time_adjusted = np.zeros((len(preds), len(times)))
            ocl_incurreds_over_actuals_by_time = np.zeros((len(incurreds), len(times)))

            # these will not be used for multiple datasets, multiple predictions but need to define them for the results dictionary
            ocl_preds_by_time = None
            ocl_incurreds_by_time = None

        preds_by_time = np.zeros((len(preds), len(times)))
        vsInc_by_time = np.zeros((len(preds), len(times)))
        weighted_vsInc_claimsize_by_time = np.zeros((len(preds), len(times)))
        weighted_vsInc_ocl_by_time = np.zeros((len(preds), len(times)))

    for index, time in enumerate(times):

        # 1 dataset, 1 prediction
        if isinstance(preds, pd.Series):
            indicator = index_data[time_str] == time

            actuals_by_time[index] = np.sum(actuals[indicator])
            incurreds_by_time[index] = np.sum(incurreds[indicator])
            ocls_by_time[index] = np.sum(ocls[indicator])
            paids_by_time[index] = actuals_by_time[index] - ocls_by_time[index]

            preds_by_time[index] = np.sum(preds[indicator])
            vsInc_by_time[index] = get_vsInc(actuals[indicator], 
                                            preds[indicator], 
                                            incurreds[indicator])
            
            weighted_vsInc_claimsize_by_time[index] = get_weighted_vsInc_claimsize(actuals[indicator], 
                                                            preds[indicator], 
                                                            incurreds[indicator])
            
            weighted_vsInc_ocl_by_time[index] = get_weighted_vsInc_ocl(actuals[indicator], 
                                                            preds[indicator], 
                                                            incurreds[indicator], 
                                                            ocls[indicator])
            
        else:
            
            # 1 dataset, multiple predictions
            if isinstance(actuals, pd.Series):
                indicator = index_data[time_str] == time       

                for i in (range(len(preds))):
                    preds_by_time[i, index] = np.sum(preds[i][indicator])
                    
                actuals_by_time[index] = np.sum(actuals[indicator])
                incurreds_by_time[index] = np.sum(incurreds[indicator])
                ocls_by_time[index] = np.sum(ocls[indicator])
                paids_by_time[index] = actuals_by_time[index] - ocls_by_time[index]

                for i in (range(len(preds))):
                    vsInc_by_time[i, index] = get_vsInc(actuals[indicator], 
                                                        preds[i][indicator], 
                                                        incurreds[indicator])

                    weighted_vsInc_claimsize_by_time[i, index] = get_weighted_vsInc_claimsize(actuals[indicator], 
                                                                        preds[i][indicator], 
                                                                        incurreds[indicator])

                    weighted_vsInc_ocl_by_time[i, index] = get_weighted_vsInc_ocl(actuals[indicator], 
                                                                        preds[i][indicator], 
                                                                        incurreds[indicator], 
                                                                        ocls[indicator])

                    ocl_preds_by_time[i, index] = preds_by_time[i][index] - paids_by_time[index]

                ocl_incurreds_by_time[index] = incurreds_by_time[index] - paids_by_time[index]

            # multiple datasets, multiple predictions
            else:
                for i in (range(len(preds))):
                    indicator = index_data[i][time_str] == time

                    preds_by_time[i, index] = np.sum(preds[i][indicator])
                    actuals_by_time[i, index] = np.sum(actuals[i][indicator])
                    incurreds_by_time[i, index] = np.sum(incurreds[i][indicator])
                    ocls_by_time[i, index] = np.sum(ocls[i][indicator])
                    paids_by_time[i, index] = actuals_by_time[i, index] - ocls_by_time[i, index]

                    vsInc_by_time[i, index] = get_vsInc(actuals[i][indicator], 
                                                    preds[i][indicator], 
                                                    incurreds[i][indicator]) 
                
                    weighted_vsInc_claimsize_by_time[i, index] = get_weighted_vsInc_claimsize(actuals[i][indicator], 
                                                                    preds[i][indicator], 
                                                                    incurreds[i][indicator])

                    weighted_vsInc_ocl_by_time[i, index] = get_weighted_vsInc_ocl(actuals[i][indicator], 
                                                                    preds[i][indicator], 
                                                                    incurreds[i][indicator], 
                                                                    ocls[i][indicator])
                    
                    preds_over_actuals_by_time[i, index] = np.sum(preds[i][indicator]) / np.sum(actuals[i][indicator]) if np.sum(actuals[i][indicator]) > 0 else 1
                    incurreds_over_actuals_by_time[i, index] = np.sum(incurreds[i][indicator]) / np.sum(actuals[i][indicator]) if np.sum(actuals[i][indicator]) > 0 else 1

                    ocl_preds_over_actuals_by_time[i, index] = (np.sum(preds[i][indicator]) - paids_by_time[i, index]) / ocls_by_time[i, index] if ocls_by_time[i, index] > 0 else 1

                    # replacing negative ocl preds with the paid amounts
                    ocl_preds_over_actuals_by_time_adjusted[i, index] = (np.sum(preds[i][(indicator) & (preds[i] - (actuals[i] - ocls[i]) > 0)]) 
                                                                         + np.sum(actuals[i][(indicator) & (preds[i] - (actuals[i] - ocls[i]) <= 0)])
                                                                         - np.sum(ocls[i][(indicator) & (preds[i] - (actuals[i] - ocls[i]) <= 0)]) 
                                                                         - paids_by_time[i, index]) / ocls_by_time[i, index] if ocls_by_time[i, index] > 0 else 1
                    
                    ocl_incurreds_over_actuals_by_time[i, index] = (np.sum(incurreds[i][indicator]) - paids_by_time[i, index]) / ocls_by_time[i, index] if ocls_by_time[i, index] > 0 else 1

                # adjusts negative ocl predictions to 0
                #ocl_preds_over_actuals_by_time_adjusted =  ocl_preds_over_actuals_by_time.copy()
                #ocl_preds_over_actuals_by_time_adjusted[ocl_preds_over_actuals_by_time_adjusted < 0] = 0

    results = {
        'actuals_by_time': actuals_by_time,
        'incurreds_by_time': incurreds_by_time,
        'ocls_by_time': ocls_by_time,
        'paids_by_time': paids_by_time,
        'preds_by_time': preds_by_time,
        'vsInc_by_time': vsInc_by_time,
        'weighted_vsInc_claimsize_by_time': weighted_vsInc_claimsize_by_time,
        'weighted_vsInc_ocl_by_time': weighted_vsInc_ocl_by_time,
        'ocl_preds_by_time': ocl_preds_by_time,
        'ocl_incurreds_by_time': ocl_incurreds_by_time,
        'preds_over_actuals_by_time': preds_over_actuals_by_time,
        'incurreds_over_actuals_by_time': incurreds_over_actuals_by_time,
        'ocl_preds_over_actuals_by_time': ocl_preds_over_actuals_by_time,
        'ocl_preds_over_actuals_by_time_adjusted': ocl_preds_over_actuals_by_time_adjusted,
        'ocl_incurreds_over_actuals_by_time': ocl_incurreds_over_actuals_by_time,
        'times': times,
        'index_data': index_data,
        'actuals': actuals,
        'preds': preds,
        'incurreds': incurreds,
        'ocls': ocls,
        'time_str': time_str
    }

    return results

def graph_by_time(results_model1, name_model1=None, results_model2=None, name_model2=None, include_incurreds=True):
    ''' results_modelx is a dictionary of the results from the model,
    include_incurreds is a boolean to include the raw case estimates in the graph
    if results_model2 is passed, then name_model1 and name_model2 need to be passed'''

    # Converting aggregate OCLs to proportion
    if not (isinstance(results_model1['preds'], pd.Series) or isinstance(results_model1['actuals'], pd.Series)):
        # multiple datasets
        # sum over multiple datasets (could change this later to have a boxplot of claim counts)
        ocls_by_time = np.sum(results_model1['ocls_by_time'], axis=0)
    
    else:
        # 1 dataset
        ocls_by_time = results_model1['ocls_by_time']

    pred_cumulative_ocl_by_time = np.cumsum(ocls_by_time)
    pred_cumulative_prop_by_time = pred_cumulative_ocl_by_time / pred_cumulative_ocl_by_time[-1]

    #print(pred_cumulative_ocl_by_time)
    #print(pred_cumulative_prop_by_time)

    times = results_model1['times']
    time_str = results_model1['time_str']
    actuals = results_model1['actuals']
    actuals_by_time = results_model1['actuals_by_time']
    paids_by_time = results_model1['paids_by_time']
    ocls_by_time = results_model1['ocls_by_time']

    if include_incurreds:
        incurreds_by_time = results_model1['incurreds_by_time']
        ocl_incurreds_by_time = results_model1['ocl_incurreds_by_time']
        incurreds_over_actuals_by_time = results_model1['incurreds_over_actuals_by_time']
        ocl_incurreds_over_actuals_by_time = results_model1['ocl_incurreds_over_actuals_by_time']

    # 1 dataset, 1 prediction
    if isinstance(results_model1['preds'], pd.Series):
        # plotting aggregate preds
        plt.plot(times, actuals_by_time, label='Actuals')
        if name_model1 is None:
            plt.plot(times, results_model1['preds_by_time'], label='Predictions')
        else:
            plt.plot(times, results_model1['preds_by_time'], label=name_model1)
        if results_model2 is not None:
            plt.plot(times, results_model2['preds_by_time'], label=name_model2)
        if include_incurreds:
            plt.plot(times, incurreds_by_time, label='Case Estimates')
        plt.legend(loc='upper right')
        plt.title('Aggregate claim sizes')

        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')
            
        plt.show()

        # plotting aggregate ocls
        plt.plot(times, ocls_by_time, label='Actuals')
        if name_model1 is None:
            plt.plot(times, results_model1['preds_by_time'] - paids_by_time, label='Predictions')
        else:
            plt.plot(times, results_model1['preds_by_time'] - paids_by_time, label=name_model1)
        if results_model2 is not None:
            plt.plot(times, results_model2['preds_by_time'] - paids_by_time, label=name_model2)
        if include_incurreds:
            plt.plot(times, incurreds_by_time - paids_by_time, label='Case Estimates')
        plt.legend(loc='upper right')
        plt.title('OCL')

        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')
            
        plt.show()

        # plotting vsInc
        if name_model1 is None:
            plt.plot(times, results_model1['vsInc_by_time'])
        else:
            plt.plot(times, results_model1['vsInc_by_time'], label=name_model1)
        if results_model2 is not None:
            plt.plot(times, results_model2['vsInc_by_time'], label=name_model2)
            plt.legend(loc='upper right')

        plt.ylabel('vsInc (%)')

        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')
            
        plt.show()

        # plotting weighted vsInc by claim size
        if name_model1 is None:
            plt.plot(times, results_model1['weighted_vsInc_claimsize_by_time'])
        else:
            plt.plot(times, results_model1['weighted_vsInc_claimsize_by_time'], label=name_model1)
        if results_model2 is not None:
            plt.plot(times, results_model2['weighted_vsInc_claimsize_by_time'], label=name_model2)
            plt.legend(loc='upper right')

        plt.ylabel('Weighted vsInc (Claim Size) (%)')

        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')
            
        plt.show()

        # plotting weighted vsInc by ocl
        if name_model1 is None:
            plt.plot(times, results_model1['weighted_vsInc_ocl_by_time'])
        else:
            plt.plot(times, results_model1['weighted_vsInc_ocl_by_time'], label=name_model1)
        if results_model2 is not None:
            plt.plot(times, results_model2['weighted_vsInc_ocl_by_time'], label=name_model2)
            plt.legend(loc='upper right')

        plt.ylabel('Weighted vsInc (OCL) (%)')

        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')
        
        plt.show()

    else:
        
        # 1 dataset, multiple predictions
        if isinstance(actuals, pd.Series):
            # Boxplot with better colours
            bp_preds_model1 = box_plot(results_model1['preds_by_time'], positions=times, model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['preds_by_time'], positions=times, model_name=name_model2)
            plt.plot(times, actuals_by_time)
            if include_incurreds:
                plt.plot(times, incurreds_by_time, color = 'green')

            if name_model1 is None and name_model2 is None:
                plt.legend([bp_preds_model1["boxes"][0]], ['Predictions'])
            else:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('Aggregate claim sizes')
            plt.ylabel('Ratio')
            plt.show()
            
            # Boxplot with aggregate OCLs instead of aggregate claim sizes
            bp_preds_model1 = box_plot(results_model1['ocl_preds_by_time'], positions=times, model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['ocl_preds_by_time'], positions=times, model_name=name_model2)
            plt.plot(times, ocls_by_time)
            if include_incurreds:
                plt.plot(times, ocl_incurreds_by_time, color = 'green')

            if name_model1 is None and name_model2 is None:
                plt.legend([bp_preds_model1["boxes"][0]], ['Predictions'])
            else:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('Aggregate OCLs')
            plt.ylabel('Ratio')
            plt.show()

        # multiple datasets, multiple predictions
        else:

            # Boxplot with better colours
            bp_preds_model1 = box_plot(results_model1['preds_over_actuals_by_time'], positions=times, model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['preds_over_actuals_by_time'], positions=times, model_name=name_model2)
            if include_incurreds:
                bp_incurreds = box_plot(incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates')
            plt.plot(times, [1] * len(times))
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

            if include_incurreds:
                if name_model1 is None and name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                elif name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                else:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
            elif name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('Aggregate claim sizes (as proportion of actual)')
            plt.ylabel('Ratio')
            plt.show()
            
            # Boxplot with aggregate OCLs instead of aggregate claim sizes
            bp_preds_model1 = box_plot(results_model1['ocl_preds_over_actuals_by_time'], positions=times, model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['ocl_preds_over_actuals_by_time'], positions=times, model_name=name_model2)
            if include_incurreds:
                bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates')
            plt.plot(times, [1] * len(times))
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

            if include_incurreds:
                if name_model1 is None and name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                elif name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                else:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
            elif name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('OCL (as proportion of actual)')
            plt.ylabel('Ratio')

            plt.ylim(0, 2.5)

            plt.show()

            # Boxplot with incurreds only
            if include_incurreds:
                bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates')
                plt.plot(times, [1] * len(times))
                plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

                plt.grid(axis='both', linestyle='--', alpha=0.7)

                ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                        or (time % 10 == 0 and len(times) >= 50)]
                plt.xticks(ticks, ticks)
                
                if time_str == 'pred_time':
                    plt.xlabel('Calendar quarter')
                elif time_str == 'dev_quarter':
                    plt.xlabel('Quarters since notification')
                elif time_str == 'rept_quarter':
                    plt.xlabel('Reported quarter')
                elif time_str == 'acc_quarter':
                    plt.xlabel('Accident quarter')
                else:
                    raise ValueError('Invalid time_str')
                
                plt.title('OCL (as proportion of actual)')
                plt.ylabel('Ratio')

                plt.ylim(0, 2.5)

                plt.show()

                if time_str == 'dev_quarter':
                    bp_incurreds = box_plot([sublist[:10] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:10], model_name='Case Estimates')
                    plt.plot(times[:10], [1] * len(times[:10]))
                    plt.plot(times[:10], pred_cumulative_prop_by_time[:10], color='black', alpha = 0.8)

                    plt.grid(axis='both', linestyle='--', alpha=0.7)
                    ticks = [int(time) for time in times[:10]]
                    plt.xticks(ticks, ticks)
                    
                    plt.xlabel('Quarters since notification')
                    plt.title('OCL (as proportion of actual)')
                    plt.ylabel('Ratio')

                    plt.ylim(0, 1.5)

                    plt.show()

                    bp_incurreds = box_plot([sublist[:16] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:16], model_name='Case Estimates')
                    plt.plot(times[:16], [1] * len(times[:16]))
                    plt.plot(times[:16], pred_cumulative_prop_by_time[:16], color='black', alpha = 0.8)

                    plt.grid(axis='both', linestyle='--', alpha=0.7)
                    ticks = [int(time) for time in times[:16]]
                    plt.xticks(ticks, ticks)
                    
                    plt.xlabel('Quarters since notification')
                    plt.title('OCL (as proportion of actual)')
                    plt.ylabel('Ratio')

                    plt.ylim(0, 1.5)

                    plt.show()

            # Boxplot with aggregate OCLs instead of aggregate claim sizes (capped at dev quarter 10)
            if time_str == 'dev_quarter':
                bp_preds_model1 = box_plot([sublist[:10] for sublist in results_model1['ocl_preds_over_actuals_by_time']], positions=times[:10], model_name=name_model1)
                if results_model2 is not None:
                    bp_preds_model2 = box_plot([sublist[:10] for sublist in results_model2['ocl_preds_over_actuals_by_time']], positions=times[:10], model_name=name_model2)
                if include_incurreds:
                    bp_incurreds = box_plot([sublist[:10] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:10], model_name='Case Estimates')
                plt.plot(times[:10], [1] * len(times[:10]))
                plt.plot(times[:10], pred_cumulative_prop_by_time[:10], color='black', alpha = 0.8)

                if include_incurreds:
                    if name_model1 is None and name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                    elif name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                    else:
                        plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
                elif name_model1 is not None and name_model2 is not None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

                plt.grid(axis='both', linestyle='--', alpha=0.7)

                ticks = [int(time) for time in times[:10]]
                plt.xticks(ticks, ticks)
                
                plt.xlabel('Quarters since notification')
                plt.title('OCL (as proportion of actual)')
                plt.ylabel('Ratio')

                plt.ylim(0, 1.5)

                plt.show()

                bp_preds_model1 = box_plot([sublist[:10] for sublist in results_model1['ocl_preds_over_actuals_by_time']], positions=times[:10], model_name=name_model1, alpha=0.5)
                if results_model2 is not None:
                    bp_preds_model2 = box_plot([sublist[:10] for sublist in results_model2['ocl_preds_over_actuals_by_time']], positions=times[:10], model_name=name_model2, alpha=0.5)
                if include_incurreds:
                    bp_incurreds = box_plot([sublist[:10] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:10], model_name='Case Estimates', alpha=0.5)
                plt.plot(times[:10], [1] * len(times[:10]))
                plt.plot(times[:10], pred_cumulative_prop_by_time[:10], color='black', alpha = 0.8)

                if include_incurreds:
                    if name_model1 is None and name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                    elif name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                    else:
                        plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
                elif name_model1 is not None and name_model2 is not None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

                plt.grid(axis='both', linestyle='--', alpha=0.7)

                ticks = [int(time) for time in times[:10]]
                plt.xticks(ticks, ticks)
                
                plt.xlabel('Quarters since notification')
                plt.title('OCL (as proportion of actual)')
                plt.ylabel('Ratio')

                plt.ylim(0, 1.5)

                plt.show()
            
            # Boxplot with aggregate OCLs instead of aggregate claim sizes (capped at dev quarter 16)
                bp_preds_model1 = box_plot([sublist[:16] for sublist in results_model1['ocl_preds_over_actuals_by_time']], positions=times[:16], model_name=name_model1)
                if results_model2 is not None:
                    bp_preds_model2 = box_plot([sublist[:16] for sublist in results_model2['ocl_preds_over_actuals_by_time']], positions=times[:16], model_name=name_model2)
                if include_incurreds:
                    bp_incurreds = box_plot([sublist[:16] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:16], model_name='Case Estimates')
                plt.plot(times[:16], [1] * len(times[:16]))
                plt.plot(times[:16], pred_cumulative_prop_by_time[:16], color='black', alpha = 0.8)

                if include_incurreds:
                    if name_model1 is None and name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                    elif name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                    else:
                        plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
                elif name_model1 is not None and name_model2 is not None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

                plt.grid(axis='both', linestyle='--', alpha=0.7)

                ticks = [int(time) for time in times[:16]]
                plt.xticks(ticks, ticks)
                
                plt.xlabel('Quarters since notification')
                plt.title('OCL (as proportion of actual)')
                plt.ylabel('Ratio')

                plt.ylim(0, 1.5)

                plt.show()

                bp_preds_model1 = box_plot([sublist[:16] for sublist in results_model1['ocl_preds_over_actuals_by_time']], positions=times[:16], model_name=name_model1, alpha=0.5)
                if results_model2 is not None:
                    bp_preds_model2 = box_plot([sublist[:16] for sublist in results_model2['ocl_preds_over_actuals_by_time']], positions=times[:16], model_name=name_model2, alpha=0.5)
                if include_incurreds:
                    bp_incurreds = box_plot([sublist[:16] for sublist in ocl_incurreds_over_actuals_by_time], positions=times[:16], model_name='Case Estimates', alpha=0.5)
                plt.plot(times[:16], [1] * len(times[:16]))
                plt.plot(times[:16], pred_cumulative_prop_by_time[:16], color='black', alpha = 0.8)

                if include_incurreds:
                    if name_model1 is None and name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                    elif name_model2 is None:
                        plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                    else:
                        plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
                elif name_model1 is not None and name_model2 is not None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

                plt.grid(axis='both', linestyle='--', alpha=0.7)

                ticks = [int(time) for time in times[:16]]
                plt.xticks(ticks, ticks)
                
                plt.xlabel('Quarters since notification')
                plt.title('OCL (as proportion of actual)')
                plt.ylabel('Ratio')

                plt.ylim(0, 1.5)

                plt.show()

            # Boxplot with aggregate OCLs and transparency for overlapping boxes
            bp_preds_model1 = box_plot(results_model1['ocl_preds_over_actuals_by_time'], positions=times, model_name=name_model1, alpha=0.5)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['ocl_preds_over_actuals_by_time'], positions=times, model_name=name_model2, alpha=0.5)
            if include_incurreds:
                bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates', alpha=0.5)
            plt.plot(times, [1] * len(times))
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

            if include_incurreds:
                if name_model1 is None and name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                elif name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                else:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
            elif name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('OCL (as proportion of actual)')
            plt.ylabel('Ratio')
            plt.show()

            # Boxplot with aggregate OCLs and side by side boxes
            bp_preds_model1 = box_plot(results_model1['ocl_preds_over_actuals_by_time'], positions=times-0.2, model_name=name_model1, widths=0.3)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['ocl_preds_over_actuals_by_time'], positions=times+0.2, model_name=name_model2, widths=0.3)
            if include_incurreds:
                bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates')
            plt.plot(times, [1] * len(times))
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

            if include_incurreds:
                if name_model1 is None and name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                elif name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                else:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
            elif name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('OCL (as proportion of actual)')
            plt.ylabel('Ratio')
            plt.show()

            # Boxplot with aggregate OCLs (hard code negative OCL preds as 0)
            bp_preds_model1 = box_plot(results_model1['ocl_preds_over_actuals_by_time_adjusted'], positions=times, model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot(results_model2['ocl_preds_over_actuals_by_time_adjusted'], positions=times, model_name=name_model2)
            if include_incurreds:
                bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, model_name='Case Estimates')
            plt.plot(times, [1] * len(times))
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)

            if include_incurreds:
                if name_model1 is None and name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Case Estimates'])
                elif name_model2 is None:
                    plt.legend([bp_preds_model1["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, 'Case Estimates'])
                else:
                    plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0], bp_incurreds["boxes"][0]], [name_model1, name_model2, 'Case Estimates'])
            elif name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)

            ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                     or (time % 10 == 0 and len(times) >= 50)]
            plt.xticks(ticks, ticks)
            
            if time_str == 'pred_time':
                plt.xlabel('Calendar quarter')
            elif time_str == 'dev_quarter':
                plt.xlabel('Quarters since notification')
            elif time_str == 'rept_quarter':
                plt.xlabel('Reported quarter')
            elif time_str == 'acc_quarter':
                plt.xlabel('Accident quarter')
            else:
                raise ValueError('Invalid time_str')
            
            plt.title('OCL (as proportion of actual) (manually adjusted negatives to 0)')
            plt.ylabel('Ratio')
            plt.show()

        # boxplot of weighted vsInc by claim size
        bp_preds_model1 = box_plot(results_model1['weighted_vsInc_claimsize_by_time'], positions=times, model_name=name_model1)
        if results_model2 is not None:
            bp_preds_model2 = box_plot(results_model2['weighted_vsInc_claimsize_by_time'], positions=times, model_name=name_model2)
        plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)

        if name_model1 is not None and name_model2 is not None:
            plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

        plt.title('Weighted vsInc (Claim Size)')
        plt.ylabel('Weighted vsInc (Claim Size) (%)')
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                    or (time % 10 == 0 and len(times) >= 50)]
        plt.xticks(ticks, ticks)
        
        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')

        plt.show()

        # boxplot of weighted vsInc by ocl
        bp_preds_model1 = box_plot(results_model1['weighted_vsInc_ocl_by_time'], positions=times, model_name=name_model1)
        if results_model2 is not None:
            bp_preds_model2 = box_plot(results_model2['weighted_vsInc_ocl_by_time'], positions=times, model_name=name_model2)
        plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)

        if name_model1 is not None and name_model2 is not None:
            plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

        plt.title('$vsCE_{OCL} (\%)$')
        #plt.ylabel('Weighted vsInc (OCL) (%)')
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                    or (time % 10 == 0 and len(times) >= 50)]
        plt.xticks(ticks, ticks)
        
        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')

        plt.show()

        # Boxplot of weighted vsInc by ocl (capped at dev quarter 10)
        if time_str == 'dev_quarter':

            bp_preds_model1 = box_plot([sublist[:10] for sublist in results_model1['weighted_vsInc_ocl_by_time']], positions=times[:10], model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot([sublist[:10] for sublist in results_model2['weighted_vsInc_ocl_by_time']], positions=times[:10], model_name=name_model2)
            plt.plot(times[:10], pred_cumulative_prop_by_time[:10] * 100, color='black', alpha = 0.7)

            if name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)
            ticks = [int(time) for time in times[:10]]
            plt.xticks(ticks, ticks)
            
            plt.xlabel('Quarters since notification')
            plt.title('$vsCE_{OCL} (\%)$')
            #plt.ylabel('Weighted vsInc (OCL) (%)')
            
            plt.show()

            bp_preds_model1 = box_plot([sublist[:10] for sublist in results_model1['weighted_vsInc_ocl_by_time']], positions=times[:10], model_name=name_model1, alpha=0.5)
            if results_model2 is not None:
                bp_preds_model2 = box_plot([sublist[:10] for sublist in results_model2['weighted_vsInc_ocl_by_time']], positions=times[:10], model_name=name_model2, alpha=0.5)
            plt.plot(times[:10], pred_cumulative_prop_by_time[:10] * 100, color='black', alpha = 0.7)

            if name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)
            ticks = [int(time) for time in times[:10]]
            plt.xticks(ticks, ticks)
            
            plt.xlabel('Quarters since notification')
            plt.title('$vsCE_{OCL} (\%)$')
            #plt.ylabel('Weighted vsInc (OCL) (%)')
            
            plt.show()

        # Boxplot of weighted vsInc by ocl (capped at dev quarter 16)
            bp_preds_model1 = box_plot([sublist[:16] for sublist in results_model1['weighted_vsInc_ocl_by_time']], positions=times[:16], model_name=name_model1)
            if results_model2 is not None:
                bp_preds_model2 = box_plot([sublist[:16] for sublist in results_model2['weighted_vsInc_ocl_by_time']], positions=times[:16], model_name=name_model2)
            plt.plot(times[:16], pred_cumulative_prop_by_time[:16] * 100, color='black', alpha = 0.7)

            if name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)
            ticks = [int(time) for time in times[:16]]
            plt.xticks(ticks, ticks)
            
            plt.xlabel('Quarters since notification')
            plt.title('$vsCE_{OCL} (\%)$')
            #plt.ylabel('Weighted vsInc (OCL) (%)')

            plt.show()

            bp_preds_model1 = box_plot([sublist[:16] for sublist in results_model1['weighted_vsInc_ocl_by_time']], positions=times[:16], model_name=name_model1, alpha=0.5)
            if results_model2 is not None:
                bp_preds_model2 = box_plot([sublist[:16] for sublist in results_model2['weighted_vsInc_ocl_by_time']], positions=times[:16], model_name=name_model2, alpha=0.5)
            plt.plot(times[:16], pred_cumulative_prop_by_time[:16] * 100, color='black', alpha = 0.7)

            if name_model1 is not None and name_model2 is not None:
                plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

            plt.grid(axis='both', linestyle='--', alpha=0.7)
            ticks = [int(time) for time in times[:16]]
            plt.xticks(ticks, ticks)
            
            plt.xlabel('Quarters since notification')
            plt.title('$vsCE_{OCL} (\%)$')
            #plt.ylabel('Weighted vsInc (OCL) (%)')

            plt.show()


        # boxplot of weighted vsInc by ocl with transparency for overlapping boxes
        bp_preds_model1 = box_plot(results_model1['weighted_vsInc_ocl_by_time'], positions=times, model_name=name_model1, alpha=0.5)
        if results_model2 is not None:
            bp_preds_model2 = box_plot(results_model2['weighted_vsInc_ocl_by_time'], positions=times, model_name=name_model2, alpha=0.5)
        plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)

        if name_model1 is not None and name_model2 is not None:
            plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

        plt.title('Weighted vsInc (OCL)')
        plt.ylabel('Weighted vsInc (OCL) (%)')
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                    or (time % 10 == 0 and len(times) >= 50)]
        plt.xticks(ticks, ticks)
        
        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')

        plt.show()

        # boxplot of weighted vsInc by ocl with side by side boxes
        bp_preds_model1 = box_plot(results_model1['weighted_vsInc_ocl_by_time'], positions=times-0.2, model_name=name_model1, widths=0.3)
        if results_model2 is not None:
            bp_preds_model2 = box_plot(results_model2['weighted_vsInc_ocl_by_time'], positions=times+0.2, model_name=name_model2, widths=0.3)
        plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)

        if name_model1 is not None and name_model2 is not None:
            plt.legend([bp_preds_model1["boxes"][0], bp_preds_model2["boxes"][0]], [name_model1, name_model2])

        plt.title('Weighted vsInc (OCL)')
        plt.ylabel('Weighted vsInc (OCL) (%)')
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        ticks = [int(time) for time in times if (time % 5 == 0 and len(times) < 50) 
                    or (time % 10 == 0 and len(times) >= 50)]
        plt.xticks(ticks, ticks)
        
        if time_str == 'pred_time':
            plt.xlabel('Calendar quarter')
        elif time_str == 'dev_quarter':
            plt.xlabel('Quarters since notification')
        elif time_str == 'rept_quarter':
            plt.xlabel('Reported quarter')
        elif time_str == 'acc_quarter':
            plt.xlabel('Accident quarter')
        else:
            raise ValueError('Invalid time_str')

        plt.show()

def aggregate_by_time(index_data, actuals, preds, incurreds, ocls, time_str, model_name=None):
    '''Plots aggregate claims, vsInc and weighted vsInc over time, 
       either calendar or development
       
       time_str should be either 'pred_time' (for calendar quarter results), 
       'dev_quarter' for 'development' quarter results (i.e. time since notification, not since occurrence),
       'acc_quarter' for accident quarter results,
       or 'rept_quarter' for reported quarter results'''

    results_model1 = extract_performance_by_time(index_data, actuals, preds, incurreds, ocls, time_str)
    graph_by_time(results_model1, name_model1=model_name, include_incurreds=True)
    graph_by_time(results_model1, name_model1=model_name, include_incurreds=False)


def get_aggregates(actuals, preds, incurreds, ocls):
    '''Prints the sum over each claim and censor point for all claims, 
       predictions and case estimates'''

    aggregate_preds = np.sum(preds)
    aggregate_actual = np.sum(actuals)
    aggregate_incurred = np.sum(incurreds)
    aggregate_ocl = np.sum(ocls)
    aggregate_payments = aggregate_actual - aggregate_ocl
    aggregate_pred_ocl = aggregate_preds - aggregate_payments
    aggregate_incurred_ocl = aggregate_incurred - aggregate_payments

    print(f'Aggregate predicted Claim Size: {aggregate_preds:,.0f}')
    print(f'Aggregate actual Claim Size: {aggregate_actual:,.0f}')
    print(f'Aggregate incurred Claim Size: {aggregate_incurred:,.0f}')
    print(f'Aggregate payments: {aggregate_payments:,.0f}')
    print(f'Aggregate predicted OCL: {aggregate_pred_ocl:,.0f}')
    print(f'Aggregate actual OCL: {aggregate_ocl:,.0f}')
    print(f'Aggregate incurred OCL: {aggregate_incurred_ocl:,.0f}')

def get_small_large(actuals, preds, incurreds, ocls, test_data, small_threshold, 
                    large_threshold):
    
    '''Splits the data into small, medium and large claims based on the 
    thresholds'''
    
    small_actuals = actuals[actuals < small_threshold]
    small_preds = preds[actuals < small_threshold]
    small_incurreds = incurreds[actuals < small_threshold]
    small_ocls = ocls[actuals < small_threshold]

    medium_actuals = actuals[(actuals > small_threshold) & 
                             (actuals < large_threshold)]
    
    medium_preds = preds[(actuals > small_threshold) & 
                         (actuals < large_threshold)]
    
    medium_incurreds = incurreds[(actuals > small_threshold) & 
                                 (actuals < large_threshold)]
    
    medium_ocls = ocls[(actuals > small_threshold) &
                       (actuals < large_threshold)]

    large_actuals = actuals[actuals >= large_threshold]
    large_preds = preds[actuals >= large_threshold]
    large_incurreds = incurreds[actuals >= large_threshold]
    large_ocls = ocls[actuals >= large_threshold]

    small_data = deepcopy(test_data)
    small_data.index = test_data.index[actuals < small_threshold]

    medium_data = deepcopy(test_data)
    medium_data.index = test_data.index[(actuals > small_threshold) & (actuals < large_threshold)]

    large_data = deepcopy(test_data)
    large_data.index = test_data.index[actuals >= large_threshold]

    return (small_actuals, small_preds, small_incurreds, small_ocls,
            medium_actuals, medium_preds, medium_incurreds, medium_ocls,
            large_actuals, large_preds, large_incurreds, large_ocls,
            small_data, medium_data, large_data)

def get_close_far(actuals, preds, incurreds):
    '''Plots the distribution of the winning and losing margins for the 
       model's predictions'''

    closer = np.abs(preds - actuals) < np.abs(incurreds - actuals)

    closer_preds = preds[closer]
    closer_actuals = actuals[closer]
    closer_incurreds = incurreds[closer]

    further_preds = preds[~closer]
    further_actuals = actuals[~closer]
    further_incurreds = incurreds[~closer]

    winning_margins = 100 * np.abs(closer_preds - closer_incurreds) / closer_actuals
    losing_margins = 100 * np.abs(further_preds - further_incurreds) / further_actuals

    # plotting each distribution separately

    if winning_margins.size != 0:
        plt.hist(winning_margins, 
                 weights=(np.zeros_like(winning_margins) + 1. / 
                          winning_margins.size), 
                 alpha=0.5, 
                 label='winning margin', 
                 color='blue')
        
        #plt.legend(loc='upper right')
        plt.xlabel('Winning Margin (%)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print('There are no winning margins')

    if losing_margins.size != 0:
        plt.hist(losing_margins, 
                 weights=(np.zeros_like(losing_margins) + 1. / 
                          losing_margins.size), 
                 alpha=0.5, 
                 label='losing margin', 
                 color='red')
        
        #plt.legend(loc='upper right')
        plt.xlabel('Losing Margin (%)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print('There are no losing margins')

    # plotting both distributions together
    if winning_margins.size != 0 and losing_margins.size != 0:

        bins = np.histogram(np.hstack((winning_margins, losing_margins)), 
                            bins=10)[1]

        plt.hist(winning_margins,
                 bins=bins,
                 alpha=0.5, 
                 label='winning margin', 
                 color='blue')
        
        plt.hist(losing_margins,
                 bins=bins, 
                 alpha=0.5, 
                 label='losing margin', 
                 color='red')
        
        plt.legend(loc='upper right')
        plt.xlabel('Margin (%)')
        plt.ylabel('Count')
        plt.show()

def analyse_model(model, dataset, hp_comb, 
                  small_threshold=40000, large_threshold=500000):
    
    ''' Generates graphical and numerical results for the specified model'''

    # Analysing all claims at all prediction times
    print('All')
    actuals, preds, incurreds, ocls = get_preds_actuals(model, dataset, hp_comb)

    # apply bias correction
    if dataset.target_col == 'log_claim_size' or dataset.target_col == 'log_true_ocl':
        preds = preds * bias_correction_factor(preds, actuals)

    get_aggregates(actuals, preds, incurreds, ocls)
    get_losses(actuals, preds, incurreds, dataset, hp_comb)
    print(f'vsInc: {get_vsInc(actuals, preds, incurreds):.2f}%')
    print(f'Weighted vsInc (Claim Size): {get_weighted_vsInc_claimsize(actuals, preds, incurreds):.2f}%')
    print(f'Weighted vsInc (OCL): {get_weighted_vsInc_ocl(actuals, preds, incurreds, ocls):.2f}%')
    print(f'number of preds: {len(preds)}')

    #print(f'preds: min = {preds.min()}, mean = {preds.mean()}, max = {preds.max()}')

    get_heatmap(actuals, preds, nbins=50)
    get_close_far(actuals, preds, incurreds)


    # Analysing the latest prediction for each claim
    print('Latest')
    (latest_actuals, latest_preds, 
     latest_incurreds, latest_ocls, latest_data) = get_latest(dataset, actuals, 
                                                 preds, incurreds, ocls)
    
    get_aggregates(latest_actuals, latest_preds, latest_incurreds, latest_ocls)
    get_losses(latest_actuals, latest_preds, latest_incurreds, latest_data, hp_comb)
    print(f'vsInc: {get_vsInc(latest_actuals, latest_preds, latest_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(latest_actuals, 
                                        latest_preds, 
                                        latest_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(latest_actuals,
                                                latest_preds,
                                                latest_incurreds,
                                                latest_ocls)
    
    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(latest_preds)}')

    get_heatmap(latest_actuals, latest_preds, nbins=40)
    get_close_far(latest_actuals, latest_preds, latest_incurreds)

    aggregate_by_time(dataset.index, actuals, preds, incurreds, ocls, 'pred_time')
    aggregate_by_time(dataset.index, actuals, preds, incurreds, ocls, 'dev_quarter')
    aggregate_by_time(dataset.index, actuals, preds, incurreds, ocls, 'rept_quarter')
    aggregate_by_time(dataset.index, actuals, preds, incurreds, ocls, 'acc_quarter')


    # Analysing all claims by the specified development quarters
    dev_quarters = [1, 5, 10, 16] # can adjust these to analyse different periods
    nbinss = [40, 30, 30, 20]
    for i in range(len(dev_quarters)):
        print(f'Dev Quarter {dev_quarters[i]}')
        (dev_actuals, dev_preds, 
         dev_incurreds, dev_ocls, dev_data) = get_dev_quarter(dataset, actuals, preds, 
                                                    incurreds, ocls, dev_quarters[i])
        
        get_aggregates(dev_actuals, dev_preds, dev_incurreds, dev_ocls)
        get_losses(dev_actuals, dev_preds, dev_incurreds, dev_data, hp_comb)
        print(f'vsInc: {get_vsInc(dev_actuals, dev_preds, dev_incurreds):.2f}%')

        weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(dev_actuals, 
                                            dev_preds, 
                                            dev_incurreds)
        
        weighted_vsinc_ocl = get_weighted_vsInc_ocl(dev_actuals,
                                                    dev_preds,
                                                    dev_incurreds,
                                                    dev_ocls)
        
        print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
        print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
        print(f'number of preds: {len(dev_preds)}')
        get_heatmap(dev_actuals, dev_preds, nbins=nbinss[i])


    # Analysing all claims by size
    (small_actuals, small_preds, small_incurreds, small_ocls,
     medium_actuals, medium_preds, medium_incurreds, medium_ocls,
     large_actuals, large_preds, large_incurreds, large_ocls,
     small_data, medium_data, 
     large_data) = get_small_large(actuals, preds, incurreds, ocls, dataset,
                                   small_threshold, large_threshold)
    
    print('Small')
    get_aggregates(small_actuals, small_preds, small_incurreds, small_ocls)
    get_losses(small_actuals, small_preds, small_incurreds, small_data, hp_comb)
    print(f'vsInc: {get_vsInc(small_actuals, small_preds, small_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(small_actuals, 
                                        small_preds, 
                                        small_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(small_actuals,
                                                small_preds,
                                                small_incurreds,
                                                small_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(small_preds)}')
    get_heatmap(small_actuals, small_preds, nbins=30)
    get_close_far(small_actuals, small_preds, small_incurreds)

    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds,
                                                           small_ocls, 
                                                           'pred_time')
    
    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds, 
                                                           small_ocls,
                                                           'dev_quarter')
    
    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds, 
                                                           small_ocls,
                                                           'rept_quarter')
    
    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds, 
                                                           small_ocls,
                                                           'acc_quarter')

    print('Medium')
    get_aggregates(medium_actuals, medium_preds, medium_incurreds, medium_ocls)
    get_losses(medium_actuals, medium_preds, medium_incurreds, medium_data, hp_comb)
    print(f'vsInc: {get_vsInc(medium_actuals, medium_preds, medium_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(medium_actuals, 
                                        medium_preds, 
                                        medium_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(medium_actuals,
                                                medium_preds,
                                                medium_incurreds,
                                                medium_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(medium_preds)}')
    get_heatmap(medium_actuals, medium_preds, nbins=40)
    get_close_far(medium_actuals, medium_preds, medium_incurreds)

    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, medium_ocls, 'pred_time')
    
    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, medium_ocls, 'dev_quarter')
    
    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, medium_ocls, 'rept_quarter')
    
    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, medium_ocls, 'acc_quarter')

    print('Large')
    get_aggregates(large_actuals, large_preds, large_incurreds, large_ocls)
    get_losses(large_actuals, large_preds, large_incurreds, large_data, hp_comb)
    print(f'vsInc: {get_vsInc(large_actuals, large_preds, large_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(large_actuals, 
                                        large_preds, 
                                        large_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(large_actuals,
                                                large_preds,
                                                large_incurreds,
                                                large_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(large_preds)}')
    get_heatmap(large_actuals, large_preds,nbins=30)
    get_close_far(large_actuals, large_preds, large_incurreds)

    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, large_ocls, 'pred_time')
    
    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, large_ocls, 'dev_quarter')

    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, large_ocls, 'rept_quarter')
    
    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, large_ocls, 'acc_quarter')

    # Analysing latest predictions by ultimate size of claim
    (small_actuals, small_preds, small_incurreds, small_ocls,
     medium_actuals, medium_preds, medium_incurreds, medium_ocls,
     large_actuals, large_preds, large_incurreds, large_ocls,
     small_data, medium_data, 
     large_data) = get_small_large(latest_actuals, latest_preds, latest_incurreds, latest_ocls,
                                   latest_data, small_threshold, large_threshold)
    
    print('Small Latest')
    get_aggregates(small_actuals, small_preds, small_incurreds, small_ocls,)
    get_losses(small_actuals, small_preds, small_incurreds, small_data, hp_comb)
    print(f'vsInc: {get_vsInc(small_actuals, small_preds, small_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(small_actuals, 
                                        small_preds, 
                                        small_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(small_actuals,
                                                small_preds,
                                                small_incurreds,
                                                small_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(small_preds)}')
    get_heatmap(small_actuals, small_preds, nbins=10)
    get_close_far(small_actuals, small_preds, small_incurreds)

    print('Medium Latest')
    get_aggregates(medium_actuals, medium_preds, medium_incurreds, medium_ocls)
    get_losses(medium_actuals, medium_preds, medium_incurreds, medium_data, hp_comb)
    print(f'vsInc: {get_vsInc(medium_actuals, medium_preds, medium_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(medium_actuals, 
                                        medium_preds, 
                                        medium_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(medium_actuals,
                                                medium_preds,
                                                medium_incurreds,
                                                medium_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(medium_preds)}')
    get_heatmap(medium_actuals, medium_preds, nbins=30)
    get_close_far(medium_actuals, medium_preds, medium_incurreds)

    print('Large Latest')
    get_aggregates(large_actuals, large_preds, large_incurreds, large_ocls)
    get_losses(large_actuals, large_preds, large_incurreds, large_data, hp_comb)
    print(f'vsInc: {get_vsInc(large_actuals, large_preds, large_incurreds):.2f}%')

    weighted_vsinc_claimsize = get_weighted_vsInc_claimsize(large_actuals, 
                                        large_preds, 
                                        large_incurreds)
    
    weighted_vsinc_ocl = get_weighted_vsInc_ocl(large_actuals,
                                                large_preds,
                                                large_incurreds,
                                                large_ocls)

    print(f'Weighted vsInc (Claim Size): {weighted_vsinc_claimsize:.2f}%')
    print(f'Weighted vsInc (OCL): {weighted_vsinc_ocl:.2f}%')
    print(f'number of preds: {len(large_preds)}')
    get_heatmap(large_actuals, large_preds, nbins=10)
    get_close_far(large_actuals, large_preds, large_incurreds)

def cross_validate(fp_in, fp_out, fp_hp_comb, hyperparameter_grid, verbose=True):
    """
    Trains each model in the grid, tunes using a validation set, chooses the 
    'best' one based on the smallest validation loss and produces numerical 
    and graphical summaries for the best model.

    Args:
    fp_in: filepath to the folder with the train and val indexes and sets
    fp_out: filepath to the csv file that stores the results
     - appends results to the csv so that multiple runs are stored in the same csv file
    hyperparameter_grid: list of dictionaries.
    - Each dictionary contains the hyperparameter as a key and the value as 
      the value to try. 
    - There will only be 1 value for every key.
    verbose: whether to print written outputs and progress to console
    model_type: the type of model to train. Either "RNN" (includes LSTM and GRU) or "FNN"

    NOTE: do not change loss function within 1 run of this function becuase 
          the loss numbers will be on a different scale.
    Same with switching from 'claim_size' to 'log_m'
    """

    best_val_loss = np.Inf
    best_val_vsInc = 0
    best_val_weighted_vsInc_claimsize = 0
    best_val_weighted_vsInc_ocl = 0
    best_val_uie = np.Inf
    best_hp_comb = None
    
    for hp_comb in hyperparameter_grid:
        if verbose:
            print(f'\nTrying hyperparameter combination: {hp_comb}')

        cv_loss_list = []
        cv_vsInc_list = []
        cv_weighted_vsInc_claimsize_list = []
        cv_weighted_vsInc_ocl_list = []
        cv_uie_list = []

        # Use the same seed for each hyperparameter combination so they 
        # all start from the same initial weights
        # Also means results can be reproduced for the best combination without
        # having to rerun the entire hyperparamter tuning process
        torch.manual_seed(1)

        train_set = ClaimsDataset(hp_comb['target_col'], 
                                  fp_in + 'train_index.csv', 
                                  fp_in + 'train_set.csv',
                                  hp_comb['include_incurreds'],
                                  hp_comb['include_covariates'],
                                  hp_comb['transform_inputs'],
                                  hp_comb['model_type'],
                                  scaler=None)
        
        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_in + 'val_index.csv', 
                                fp_in + 'val_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=train_set.scaler)
        
        if hp_comb['model_type'] == "RNN":
            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                          hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                          hp_comb['output_layer'], hp_comb['dropout'], 
                          hp_comb['normalisation'], hp_comb['include_incurreds'], 
                          hp_comb['include_covariates']).to(device)
            
        elif hp_comb['model_type'] == "FNN":
            model = ClaimsFNN(hp_comb['nLayers'], hp_comb['nHidden'],  
                              hp_comb['dropout'], hp_comb['output_layer'],
                              hp_comb['normalisation'],
                              hp_comb['include_incurreds'], 
                              hp_comb['include_covariates']).to(device)
            
        else:
            ValueError('Invalid model type. Must be "RNN" or "FNN"')

        # Apply weight initialization
        initialise_weights(model)

        print(f'number of trainable parameters: {get_model_params_num(model, trainable_only=True)}')
        print(f'number of total parameters: {get_model_params_num(model, trainable_only=False)}')

        train_network(model, train_set, hp_comb, verbose, val_set, 
                      cv_loss_list, cv_vsInc_list, 
                      cv_weighted_vsInc_claimsize_list, 
                      cv_weighted_vsInc_ocl_list, cv_uie_list)
        
        cv_loss = np.mean(cv_loss_list)
        cv_vsInc = np.mean(cv_vsInc_list)
        cv_weighted_vsInc_claimsize = np.mean(cv_weighted_vsInc_claimsize_list)
        cv_weighted_vsInc_ocl = np.mean(cv_weighted_vsInc_ocl_list)
        cv_uie = np.mean(cv_uie_list)

        # 'best' model chosen based on validation vsInc (OCL)
        # used to be based on val loss, but vsInc is a better reflection of what we want our model to focus on
        if cv_weighted_vsInc_ocl > best_val_weighted_vsInc_ocl:
            best_val_loss = cv_loss
            best_val_vsInc = cv_vsInc
            best_val_weighted_vsInc_claimsize = cv_weighted_vsInc_claimsize
            best_val_weighted_vsInc_ocl = cv_weighted_vsInc_ocl
            best_val_uie = cv_uie
            best_hp_comb = hp_comb
            best_weights = deepcopy(model.state_dict())
            print(f'\nnew best val_loss: {round_threshold(best_val_loss):,}, '
                  f'val_vsInc: {best_val_vsInc:.2f}%, '
                  f'val_weighted_vsInc_claimsize: {best_val_weighted_vsInc_claimsize:.2f}%, '
                  f'val_weighted_vsInc_ocl: {best_val_weighted_vsInc_ocl:.2f}%, '
                  f'val_uie = {best_val_uie:.2f}%\n')
        
        # appending results to dataframe
        row = pd.DataFrame({'dataset': fp_in.split('/')[-2], 
                            'model_type': hp_comb['model_type'],
                            'include_incurreds': hp_comb['include_incurreds'],
                            'include_covariates': hp_comb['include_covariates'],
                            'transform_inputs': hp_comb['transform_inputs'],
                            'version_no': 1, 
                            'target_col': hp_comb['target_col'], 
                            'type': hp_comb['type'], 
                            'nLayers': hp_comb['nLayers'], 
                            'nHidden': hp_comb['nHidden'], 
                            'criterion': hp_comb['criterion'], 
                            'nonlinearity': hp_comb['nonlinearity'], 
                            'output_layer': hp_comb['output_layer'], 
                            'epochs': hp_comb['epochs'], 
                            'patience': hp_comb['patience'], 
                            'batch_size': hp_comb['batch_size'], 
                            'optimiser': hp_comb['optimiser'],
                            'learning_rate': hp_comb['lr'], 
                            'normalisation': hp_comb['normalisation'], 
                            'dropout': hp_comb['dropout'], 
                            'loss': round_threshold(cv_loss), 
                            'vsInc': round(cv_vsInc, 2), 
                            'weighted_vsInc_claimsize': round(cv_weighted_vsInc_claimsize, 2),
                            'weighted_vsInc_ocl': round(cv_weighted_vsInc_ocl, 2),
                            'UIE': round(cv_uie, 2)}, index=[0])
        
        row.to_csv(fp_out, mode='a', header=False, index=False)

    if verbose:
        print(f'\nBest hyperparameter combination: {best_hp_comb}')
        print(f'Best validation loss: {round_threshold(best_val_loss):,}')
        print(f'Best validation vsInc: {best_val_vsInc:.2f}%')
        print(f'Best validation weighted vsInc (Claim Size): {best_val_weighted_vsInc_claimsize:.2f}%')
        print(f'Best validation weighted vsInc (OCL): {best_val_weighted_vsInc_ocl:.2f}%')
        print(f'Best validation UIE: {best_val_uie:.2f}%')

    # saving hyperparameters to json file
    save_dictionary(best_hp_comb, fp_hp_comb)

    # Results for best model
    if hp_comb['model_type'] == "RNN":
        model = ClaimsRNN(best_hp_comb['nHidden'], best_hp_comb['nLayers'], 
                          best_hp_comb['nOut'], best_hp_comb['type'], 
                          best_hp_comb['nonlinearity'], 
                          best_hp_comb['output_layer'], best_hp_comb['dropout'], 
                          best_hp_comb['normalisation'], 
                          best_hp_comb['include_incurreds'], 
                          best_hp_comb['include_covariates']).to(device)

    elif hp_comb['model_type'] == "FNN":
        model = ClaimsFNN(best_hp_comb['nLayers'], best_hp_comb['nHidden'], 
                          best_hp_comb['dropout'], best_hp_comb['output_layer'],
                          best_hp_comb['normalisation'],
                          best_hp_comb['include_incurreds'], 
                          best_hp_comb['include_covariates']).to(device)

    else:
        ValueError('Invalid model type. Must be "RNN" or "FNN"')
    
    model.load_state_dict(best_weights)
    analyse_model(model, val_set, best_hp_comb)

def train_multiple_initialisations(fp_in, fp_out, iterations, verbose=True):
    '''Retrains the model on the test set multiple times, producing graphical 
       summaries for the first iteration as well as some graphical summaries 
       of the distribution of predictions
       
       Args:
         fp_in: filepath to the folder with the test indexes and sets
         fp_out: filepath to the folder that stores the trained model weights
         hp_comb: dictionary of hyperparameters
         iterations: number of times to retrain the model
         verbose: whether to print written outputs and progress to console'''
    
    hp_comb = load_dictionary(fp_out)

    train_set = ClaimsDataset(hp_comb['target_col'], 
                              fp_in + 'train_index.csv', 
                              fp_in + 'train_set.csv', 
                              hp_comb['include_incurreds'],
                              hp_comb['include_covariates'],
                              hp_comb['transform_inputs'],
                              hp_comb['model_type'],
                              scaler=None)

    val_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'val_index.csv', 
                            fp_in + 'val_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'],
                            scaler=train_set.scaler)

    for i in range(iterations):
        print(f'Iteration {i}')

        if hp_comb['model_type'] == "RNN":
            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                                hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                                hp_comb['output_layer'], hp_comb['dropout'], 
                                hp_comb['normalisation'], hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)
        
        elif hp_comb['model_type'] == "FNN":
            model = ClaimsFNN(hp_comb['nLayers'], hp_comb['nHidden'], 
                                hp_comb['dropout'], hp_comb['output_layer'],
                                hp_comb['normalisation'],
                                hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)

        else:
            ValueError('Invalid model type. Must be "RNN" or "FNN"')

        # Apply weight initialization
        initialise_weights(model)

        train_network(model, train_set, hp_comb, verbose, val_set)
            
        torch.save(model.state_dict(), fp_out + 'seed ' + fp_in.split('_')[-1][:-1] + ' run ' + str(i) + '.pt')

def test_multiple_initialisations(fp_in, fp_out, iterations, verbose=True, model_name=None):

    hp_comb = load_dictionary(fp_out)

    train_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'train_index.csv', 
                            fp_in + 'train_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'],
                            scaler=None)
    
    val_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'val_index.csv', 
                            fp_in + 'val_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'],
                            scaler=train_set.scaler)

    test_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'test_index.csv', 
                            fp_in + 'test_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'],
                            scaler=train_set.scaler)

    results = []
    val_date = 40

    for i in range(iterations):
        print(f'Iteration {i}')

        if hp_comb['model_type'] == "RNN":
            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                                hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                                hp_comb['output_layer'], hp_comb['dropout'], 
                                hp_comb['normalisation'], hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)
        
        elif hp_comb['model_type'] == "FNN":
            model = ClaimsFNN(hp_comb['nLayers'], hp_comb['nHidden'], 
                                hp_comb['dropout'], hp_comb['output_layer'],
                                hp_comb['normalisation'],
                                hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)

        else:
            ValueError('Invalid model type. Must be "RNN" or "FNN"')

        # Load trained weights
        model.load_state_dict(torch.load(fp_out + 'seed ' + fp_in.split('_')[-1][:-1] + ' run ' + str(i) + '.pt'))

        # testing the model somehow
        actuals, preds, incurreds, ocls = get_preds_actuals(model, test_set, 
                                                          hp_comb, verbose)
        
        # apply bias correction
        if val_set.target_col == 'log_claim_size' or val_set.target_col == 'log_true_ocl':
            (actuals_validation, 
             preds_validation, 
             incurreds_validation, 
             ocls_validation) = get_preds_actuals(model, val_set, hp_comb, verbose)

            preds = preds * bias_correction_factor(preds_validation, actuals_validation)
        
        paids = actuals - ocls

        vsInc = get_vsInc(actuals, preds, incurreds)
        weighted_vsInc_claimsize = get_weighted_vsInc_claimsize(actuals, preds, incurreds)
        weighted_vsInc_ocl = get_weighted_vsInc_ocl(actuals, preds, incurreds, ocls)

        preds_val = preds[test_set.index['pred_time'] == val_date]
        actuals_val = actuals[test_set.index['pred_time'] == val_date]
        incurreds_val = incurreds[test_set.index['pred_time'] == val_date]
        ocls_val = ocls[test_set.index['pred_time'] == val_date]
        paids_val = actuals_val - ocls_val

        ocl_preds_val = preds_val - paids_val
        weighted_vsInc_claimsize_val = get_weighted_vsInc_claimsize(actuals_val,
                                                                   preds_val,
                                                                   incurreds_val)
        
        weighted_vsInc_ocl_val = get_weighted_vsInc_ocl(actuals_val,
                                                        preds_val,
                                                        incurreds_val,
                                                        ocls_val)
        
        # MALE and MSLE (in terms of OCL instead of ultimate claim size)
        MALE_preds = MeanAbsoluteLogError()(preds - paids, ocls)
        MSLE_preds = MeanSquaredLogError()(preds - paids, ocls)

        MALE_preds_val = MeanAbsoluteLogError()(preds_val - paids_val, ocls_val)
        MSLE_preds_val = MeanSquaredLogError()(preds_val - paids_val, ocls_val)

        results.append({'preds': preds,
                        'preds_val': preds_val,
                        'vsInc': vsInc,
                        'weighted_vsInc_claimsize': weighted_vsInc_claimsize,
                        'weighted_vsInc_ocl': weighted_vsInc_ocl,
                        'ocl_preds_val': ocl_preds_val, 
                        'weighted_vsInc_claimsize_val': weighted_vsInc_claimsize_val,
                        'weighted_vsInc_ocl_val': weighted_vsInc_ocl_val,
                        'MALE_preds': MALE_preds,
                        'MSLE_preds': MSLE_preds,
                        'MALE_preds_val': MALE_preds_val,
                        'MSLE_preds_val': MSLE_preds_val})

    results = pd.DataFrame(results)
    #print(results)

    # formatting data for graphs by time
    preds_matrix = results['preds'].tolist()
    preds_matrix_val = results['preds_val'].tolist()

    # manually capping claim size predictions
    # largest in one of the datasets was $6m, so setting cap at $100m
    #preds_matrix[preds_matrix > 1e8] = 1e8

    # Assessing distribution of aggregate claims
    aggregate_preds = np.array(preds_matrix).sum(axis=1)

    # finding aggregate incurred at the valuation date (calendar quarter 40)
    actuals_val = actuals[test_set.index['pred_time'] == val_date]
    incurreds_val = incurreds[test_set.index['pred_time'] == val_date]
    ocls_val = ocls[test_set.index['pred_time'] == val_date]
    paids_val = actuals_val - ocls_val

    aggregate_preds_val = np.array(preds_matrix_val).sum(axis=1)

    aggregate_actuals_val = actuals_val.sum()
    aggregate_incurreds_val = incurreds_val.sum()
    aggregate_ocls_val = ocls_val.sum()
    aggregate_paids_val = paids_val.sum()

    ocl_preds = aggregate_preds_val - [aggregate_paids_val] * len(aggregate_preds_val)
    ocl_incurreds = aggregate_incurreds_val - aggregate_paids_val


    # PLOTTING RESULTS ACROSS ALL WEIGHT INITIALISATIONS

    # Plotting aggregate claim size by dev quarter and cal quarter (mean across all iterations)
    print('\nAll Predictions:\n')
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'dev_quarter', model_name)
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'pred_time', model_name)
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'rept_quarter', model_name)
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'acc_quarter', model_name)


    # Valuation date only
    print('\nValuation Date Predictions:\n')
    aggregate_by_time(test_set.index, actuals_val, preds_matrix_val, incurreds_val, ocls_val, 'dev_quarter', model_name)
    aggregate_by_time(test_set.index, actuals_val, preds_matrix_val, incurreds_val, ocls_val, 'pred_time', model_name)
    aggregate_by_time(test_set.index, actuals_val, preds_matrix_val, incurreds_val, ocls_val, 'rept_quarter', model_name)
    aggregate_by_time(test_set.index, actuals_val, preds_matrix_val, incurreds_val, ocls_val, 'acc_quarter', model_name)

    # Histogram of aggregate claims across all prediction times
    plt.hist(aggregate_preds, weights=(np.zeros_like(aggregate_preds) + 1. / 
                                       aggregate_preds.size), color='thistle')
    

    plt.xlabel('Aggregate predictions')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of vsInc accuracy
    plt.hist(results['vsInc'], weights=(np.zeros_like(results['vsInc']) + 1. / 
                                  results['vsInc'].size), color='lightgreen')
    
    plt.xlabel('vsInc accuracy (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of weighted vsInc (Claim Size)
    plt.hist(results['weighted_vsInc_claimsize'], weights=(np.zeros_like(results['weighted_vsInc_claimsize']) + 
                                           1. / results['weighted_vsInc_claimsize'].size), 
                                           color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of weighted vsInc (OCL)
    plt.hist(results['weighted_vsInc_ocl'], weights=(np.zeros_like(results['weighted_vsInc_ocl']) +
                                             1. / results['weighted_vsInc_ocl'].size),
                                                color='lightgreen')
    
    plt.xlabel('weighted vsInc (OCL) (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of aggregate claims at the valuation date
    plt.hist(aggregate_preds_val, weights=(np.zeros_like(aggregate_preds_val) + 1. / 
                                       aggregate_preds_val.size), color='thistle')
    
    plt.axvline(aggregate_actuals_val, color='dodgerblue', linestyle='dashed', 
                linewidth=2)
    
    plt.axvline(aggregate_incurreds_val, color='red', linestyle='dashed', 
                linewidth=2)

    plt.xlabel('Aggregate predictions at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of weighted vsInc (Claim Size) at the valuation date
    plt.hist(results['weighted_vsInc_claimsize_val'], 
             weights=(np.zeros_like(results['weighted_vsInc_claimsize_val']) + 
                      1. / results['weighted_vsInc_claimsize_val'].size), 
             color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of weighted vsInc (OCL) at the valuation date
    plt.hist(results['weighted_vsInc_ocl_val'],
                weights=(np.zeros_like(results['weighted_vsInc_ocl_val']) +
                            1. / results['weighted_vsInc_ocl_val'].size),
                color='lightgreen')
    
    plt.xlabel('weighted vsInc (OCL) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of OCL at the valuation date
    plt.hist(ocl_preds, weights=(np.zeros_like(ocl_preds) + 1. / 
                                       ocl_preds.size), color='thistle')
    
    plt.axvline(aggregate_ocls_val, color='dodgerblue', linestyle='dashed', 
                linewidth=2)
    
    plt.axvline(ocl_incurreds, color='red', linestyle='dashed', 
                linewidth=2)

    plt.xlabel('OCL at valuation date')
    plt.ylabel('Frequency')
    plt.show()



def train_multiple_datasets(fp_py, fp_out, seed_base, max_iter):

    hp_comb = load_dictionary(fp_out)

    for i in range(1, max_iter + 1):
        torch.manual_seed(1)
        fp_py_full = fp_py + str(i + seed_base) + '/'

        print('Seed: ' + str(i + seed_base))

        train_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'train_index.csv', 
                                fp_py_full + 'train_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=None)

        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'val_index.csv', 
                                fp_py_full + 'val_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=train_set.scaler)

        if hp_comb['model_type'] == 'RNN':
            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                                hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                                hp_comb['output_layer'], hp_comb['dropout'], 
                                hp_comb['normalisation'], hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)
                
        elif hp_comb['model_type'] == 'FNN':
            model = ClaimsFNN(hp_comb['nLayers'], hp_comb['nHidden'], 
                                hp_comb['dropout'], hp_comb['output_layer'],
                                hp_comb['normalisation'],
                                hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)

        else:
            ValueError('Invalid model type. Must be "RNN" or "FNN"')

        initialise_weights(model)

        train_network(model, train_set, hp_comb, True, val_set)

        torch.save(model.state_dict(), fp_out + 'seed ' + str(i + seed_base) + '.pt')

def results_multiple_datasets(fp_py, fp_out, seed_base, max_iter):

    hp_comb = load_dictionary(fp_out)

    results = []

    for i in range(1, max_iter + 1):
        fp_py_full = fp_py + str(i + seed_base) + '/'

        train_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'train_index.csv', 
                                fp_py_full + 'train_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=None)
        
        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'val_index.csv', 
                                fp_py_full + 'val_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=train_set.scaler)

        test_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'test_index.csv', 
                                fp_py_full + 'test_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'],
                                scaler=train_set.scaler)

        if hp_comb['model_type'] == 'RNN':
            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                                hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                                hp_comb['output_layer'], hp_comb['dropout'], 
                                hp_comb['normalisation'], hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)
                
        elif hp_comb['model_type'] == 'FNN':
            model = ClaimsFNN(hp_comb['nLayers'], hp_comb['nHidden'], 
                                hp_comb['dropout'], hp_comb['output_layer'],
                                hp_comb['normalisation'],
                                hp_comb['include_incurreds'], 
                                hp_comb['include_covariates']).to(device)

        else:
            ValueError('Invalid model type. Must be "RNN" or "FNN"')

        # load saved weights
        model.load_state_dict(torch.load(fp_out + 'seed ' + str(i + seed_base) + '.pt'))

        actuals, preds, incurreds, ocls = get_preds_actuals(model, test_set, hp_comb, True)

        paids = actuals - ocls

        # apply bias correction
        if val_set.target_col == 'log_claim_size' or val_set.target_col == 'log_true_ocl':
            (actuals_validation, 
             preds_validation, 
             incurreds_validation, 
             ocls_validation) = get_preds_actuals(model, val_set, hp_comb, True)

            preds = preds * bias_correction_factor(preds_validation, actuals_validation)
            
        vsInc = get_vsInc(actuals, preds, incurreds)    
        weighted_vsInc_claimsize = get_weighted_vsInc_claimsize(actuals, preds, incurreds)
        weighted_vsInc_ocl = get_weighted_vsInc_ocl(actuals, preds, incurreds, ocls)

        val_date = 40

        # finding aggregate incurred at the valuation date (calendar quarter 40)
        test_set_index_val = test_set.index[test_set.index['pred_time'] == val_date]
        preds_val = preds[test_set.index['pred_time'] == val_date]
        actuals_val = actuals[test_set.index['pred_time'] == val_date]
        incurreds_val = incurreds[test_set.index['pred_time'] == val_date]
        ocls_val = ocls[test_set.index['pred_time'] == val_date]

        paids_val = actuals_val - ocls_val

        aggregate_preds_val = preds_val.sum()
        aggregate_actuals_val = actuals_val.sum()
        aggregate_incurreds_val = incurreds_val.sum()
        aggregate_ocls_val = ocls_val.sum()
        aggregate_paids_val = aggregate_actuals_val - aggregate_ocls_val

        ocl_preds_val = aggregate_preds_val - aggregate_paids_val
        ocl_incurreds_val = aggregate_incurreds_val - aggregate_paids_val

        ocl_error_preds = round_threshold(100 * (ocl_preds_val - aggregate_ocls_val) / aggregate_ocls_val)
        ocl_error_incurreds = round_threshold(100 * (ocl_incurreds_val - aggregate_ocls_val) / aggregate_ocls_val)

        # weighted vsInc at the valuation date
        weighted_vsInc_claimsize_val = round_threshold(get_weighted_vsInc_claimsize(actuals_val, preds_val, incurreds_val))
        weighted_vsInc_ocl_val = round_threshold(get_weighted_vsInc_ocl(actuals_val, preds_val, incurreds_val, ocls_val))

        # MALE and MSLE (in terms of OCL instead of ultimate claim size)
        MALE_preds = MeanAbsoluteLogError()(preds - paids, ocls)
        MSLE_preds = MeanSquaredLogError()(preds - paids, ocls)

        MALE_preds_val = MeanAbsoluteLogError()(preds_val - paids_val, ocls_val)
        MSLE_preds_val = MeanSquaredLogError()(preds_val - paids_val, ocls_val)
        
        MALE_incurreds = MeanAbsoluteLogError()(incurreds - paids, ocls)
        MSLE_incurreds = MeanSquaredLogError()(incurreds - paids, ocls)

        MALE_incurreds_val = MeanAbsoluteLogError()(incurreds_val - paids_val, ocls_val)
        MSLE_incurreds_val = MeanSquaredLogError()(incurreds_val - paids_val, ocls_val)

        results.append({'Seed': i + seed_base, 
                        'test_set_index': test_set.index,
                        'test_set_index_val': test_set_index_val,
                        'actuals': actuals,
                        'preds': preds,
                        'incurreds': incurreds,
                        'ocls': ocls,
                        'actuals_val': actuals_val,
                        'preds_val': preds_val,
                        'incurreds_val': incurreds_val,
                        'ocls_val': ocls_val,
                        'vsInc': vsInc,
                        'weighted_vsInc_claimsize': weighted_vsInc_claimsize,
                        'weighted_vsInc_ocl': weighted_vsInc_ocl,
                        'aggregate_ocls_val': aggregate_ocls_val, 
                        'ocl_preds_val': ocl_preds_val, 
                        'ocl_incurreds_val': ocl_incurreds_val, 
                        'ocl_error_preds_val': ocl_error_preds, 
                        'ocl_error_incurreds_val': ocl_error_incurreds, 
                        'weighted_vsInc_claimsize_val': weighted_vsInc_claimsize_val,
                        'weighted_vsInc_ocl_val': weighted_vsInc_ocl_val,
                        'MALE_preds': MALE_preds,
                        'MSLE_preds': MSLE_preds,
                        'MALE_preds_val': MALE_preds_val,
                        'MSLE_preds_val': MSLE_preds_val,
                        'MALE_incurreds': MALE_incurreds,
                        'MSLE_incurreds': MSLE_incurreds,
                        'MALE_incurreds_val': MALE_incurreds_val,
                        'MSLE_incurreds_val': MSLE_incurreds_val})

    results = pd.DataFrame(results)
    return results

def plot_results_multiple_datasets(results, name_model1):
    # formatting data for graphs by time
    preds_matrix = results['preds'].tolist()
    actuals_matrix = results['actuals'].tolist()
    incurreds_matrix = results['incurreds'].tolist()
    ocls_matrix = results['ocls'].tolist()

    preds_matrix_val = results['preds_val'].tolist()
    actuals_matrix_val = results['actuals_val'].tolist()
    incurreds_matrix_val = results['incurreds_val'].tolist()
    ocls_matrix_val = results['ocls_val'].tolist()

    #print(f'preds_matrix: {preds_matrix}')

    print('\nAll predictions:\n')
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'dev_quarter', name_model1)
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'pred_time', name_model1)
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'rept_quarter', name_model1)
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'acc_quarter', name_model1)

    print('\nValuation Date Predictions:\n')
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'dev_quarter', name_model1)
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'pred_time', name_model1)
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'rept_quarter', name_model1)
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'acc_quarter', name_model1)


    # Histogram of distribution of vsInc accuracy
    plt.hist(results['vsInc'], weights=(np.zeros_like(results['vsInc']) + 1. / 
                                  results['vsInc'].size), color='lightgreen')
    
    plt.xlabel('vsInc accuracy (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of vsInc accuracy
    bp_preds_model1 = box_plot(results['vsInc'], [0], model_name=name_model1, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('vsInc accuracy (%)')
    plt.title('vsInc accuracy across multiple datasets')
    plt.show()

    # Histogram of distribution of weighted vsInc (Claim Size)
    plt.hist(results['weighted_vsInc_claimsize'], weights=(np.zeros_like(results['weighted_vsInc_claimsize']) + 
                                           1. / results['weighted_vsInc_claimsize'].size), 
                                           color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of weighted vsInc (Claim Size)
    bp_preds_model1 = box_plot(results['weighted_vsInc_claimsize'], [0], model_name=name_model1, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vsInc (Claim Size) (%)')
    plt.title('weighted vsInc (Claim Size) across multiple datasets')
    plt.show()

    # Histogram of distribution of weighted vsInc (OCL)
    plt.hist(results['weighted_vsInc_ocl'], weights=(np.zeros_like(results['weighted_vsInc_ocl']) +
                                                1. / results['weighted_vsInc_ocl'].size),
                                                color='lightgreen')
    
    plt.xlabel('weighted vsInc (OCL) (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of weighted vsInc (OCL)
    bp_preds_model1 = box_plot(results['weighted_vsInc_ocl'], [0], model_name=name_model1, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vsInc (OCL) (%)')
    plt.title('weighted vsInc (OCL) across multiple datasets')
    plt.show()

    # Histogram of weighted vsInc (Claim Size) at the valuation date
    plt.hist(results['weighted_vsInc_claimsize_val'], 
             weights=(np.zeros_like(results['weighted_vsInc_claimsize_val']) + 
                      1. / results['weighted_vsInc_claimsize_val'].size), 
             color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of weighted vsInc (Claim Size) at the valuation date
    bp_preds_model1 = box_plot(results['weighted_vsInc_claimsize_val'], [0], model_name=name_model1, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vsInc (Claim Size) (%) at valuation date')
    plt.title('weighted vsInc (Claim Size) at valuation date')
    plt.show()

    # Histogram of weighted vsInc (OCL) at the valuation date
    plt.hist(results['weighted_vsInc_ocl_val'],
                weights=(np.zeros_like(results['weighted_vsInc_ocl_val']) +
                            1. / results['weighted_vsInc_ocl_val'].size),
                color='lightgreen')
    
    plt.xlabel('weighted vsInc (OCL) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of weighted vsInc (OCL) at the valuation date
    fig = plt.figure(figsize=(1.2, 6.4))
    bp_preds_model1 = box_plot(results['weighted_vsInc_ocl_val'], [0], model_name=name_model1, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.ylabel('weighted vsInc (OCL) (%)')
    plt.title('$vsCE_{OCL} (\%)$')
    if name_model1 is not None:
        plt.xticks([0], [name_model1])
    plt.show()

    # Boxplots of OCL errors at valuation date
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results['ocl_error_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_incurreds = box_plot(results['ocl_error_incurreds_val'], [1], model_name='Case Estimates', showfliers=True)
    plt.axhline(0, color='black', linestyle='dashed', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if name_model1 is not None:
        plt.xticks([0, 1], [name_model1, 'Case Estimates'])
    else:
        plt.xticks([0, 1], ['Predictions', 'Case Estimates'])
    #plt.ylabel('OCL error (%)')
    plt.title('$OCLerr (\%)$')
    plt.show()

    # Histogram of MALE and MSLE
    plt.hist(results['MALE_preds'], weights=(np.zeros_like(results['MALE_preds']) + 1. / results['MALE_preds'].size), color='skyblue')
    plt.xlabel('MALE')
    plt.ylabel('Frequency')
    plt.title('MALE across multiple datasets')
    plt.show()

    plt.hist(results['MSLE_preds'], weights=(np.zeros_like(results['MSLE_preds']) + 1. / results['MSLE_preds'].size), color='sandybrown')
    plt.xlabel('MSLE')
    plt.ylabel('Frequency')
    plt.title('MSLE across multiple datasets')
    plt.show()

    # Boxplots of MALE and MSLE
    bp_preds_model1 = box_plot(results['MALE_preds'], [0], model_name=name_model1, showfliers=True)
    bp_incurreds = box_plot(results['MALE_incurreds'], [1], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if name_model1 is not None:
        plt.xticks([0, 1], [name_model1, 'Case Estimates'])
    else:
        plt.xticks([0, 1], ['Predictions', 'Case Estimates'])
    plt.title('MALE across multiple datasets')
    plt.ylabel('MALE')
    plt.show()

    bp_preds_model1 = box_plot(results['MSLE_preds'], [0], model_name=name_model1, showfliers=True)
    bp_incurreds = box_plot(results['MSLE_incurreds'], [1], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if name_model1 is not None:
        plt.xticks([0, 1], [name_model1, 'Case Estimates'])
    else:
        plt.xticks([0, 1], ['Predictions', 'Case Estimates'])
    plt.title('MSLE across multiple datasets')
    plt.ylabel('MSLE')
    plt.show()

    # Boxplots of MALE and MSLE at valuation date
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results['MALE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_incurreds = box_plot(results['MALE_incurreds_val'], [1], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if name_model1 is not None:
        plt.xticks([0, 1], [name_model1, 'Case Estimates'])
    else:
        plt.xticks([0, 1], ['Predictions', 'Case Estimates'])
    plt.title('MALE')
    #plt.ylabel('MALE')
    plt.show()

    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results['MSLE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_incurreds = box_plot(results['MSLE_incurreds_val'], [1], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if name_model1 is not None:
        plt.xticks([0, 1], [name_model1, 'Case Estimates'])
    else:
        plt.xticks([0, 1], ['Predictions', 'Case Estimates'])
    plt.title('MSLE')
    #plt.ylabel('MSLE')
    plt.show()

def test_multiple_datasets(fp_py, fp_out, seed_base, max_iter, name_model1=None):
    results = results_multiple_datasets(fp_py, fp_out, seed_base, max_iter)
    plot_results_multiple_datasets(results, name_model1)

def plot_multiple_models_by_time(results_model1, results_model2, name_model1, name_model2):

    # formatting data for graphs by time
    actuals_matrix = results_model1['actuals'].tolist()
    incurreds_matrix = results_model1['incurreds'].tolist()
    ocls_matrix = results_model1['ocls'].tolist()

    actuals_matrix_val = results_model1['actuals_val'].tolist()
    incurreds_matrix_val = results_model1['incurreds_val'].tolist()
    ocls_matrix_val = results_model1['ocls_val'].tolist()

    preds_matrix_model1 = results_model1['preds'].tolist()
    preds_matrix_val_model1 = results_model1['preds_val'].tolist()
    preds_matrix_model2 = results_model2['preds'].tolist()
    preds_matrix_val_model2 = results_model2['preds_val'].tolist()

    print('\nALL PREDICTIONS:\n')
    for time_str in ['dev_quarter', 'pred_time', 'rept_quarter', 'acc_quarter']:
        aggregate_performance1 = extract_performance_by_time(results_model1['test_set_index'], 
                                                             actuals_matrix, 
                                                             preds_matrix_model1, 
                                                             incurreds_matrix, 
                                                             ocls_matrix, 
                                                             time_str)

        aggregate_performance2 = extract_performance_by_time(results_model2['test_set_index'],
                                                             actuals_matrix, 
                                                             preds_matrix_model2, 
                                                             incurreds_matrix, 
                                                             ocls_matrix, 
                                                             time_str)
        
        graph_by_time(results_model1=aggregate_performance1,
                      results_model2=aggregate_performance2, 
                      name_model1=name_model1, 
                      name_model2=name_model2,
                      include_incurreds=True)
        
    print('\nVALUATION DATE PREDICTIONS:\n')
    for time_str in ['dev_quarter', 'pred_time', 'rept_quarter', 'acc_quarter']:
        aggregate_performance1 = extract_performance_by_time(results_model1['test_set_index_val'], 
                                                             actuals_matrix_val, 
                                                             preds_matrix_val_model1, 
                                                             incurreds_matrix_val, 
                                                             ocls_matrix_val, 
                                                             time_str)

        aggregate_performance2 = extract_performance_by_time(results_model2['test_set_index_val'], 
                                                             actuals_matrix_val, 
                                                             preds_matrix_val_model2, 
                                                             incurreds_matrix_val, 
                                                             ocls_matrix_val, 
                                                             time_str)
        
        graph_by_time(results_model1=aggregate_performance1,
                      results_model2=aggregate_performance2, 
                      name_model1=name_model1, 
                      name_model2=name_model2,
                      include_incurreds=True)    
        
        graph_by_time(results_model1=aggregate_performance1,
                      results_model2=aggregate_performance2, 
                      name_model1=name_model1, 
                      name_model2=name_model2,
                      include_incurreds=False)    

def test_multiple_models_multiple_datasets(fp_py, fp_out_model1, fp_out_model2, seed_base, max_iter, name_model1, name_model2):

    results_model1 = results_multiple_datasets(fp_py, fp_out_model1, seed_base, max_iter)
    results_model2 = results_multiple_datasets(fp_py, fp_out_model2, seed_base, max_iter)

    plot_multiple_models_by_time(results_model1, results_model2, name_model1, name_model2)

    # Boxplot of weighted vsInc (OCL)
    bp_preds_model1 = box_plot(results_model1['weighted_vsInc_ocl'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['weighted_vsInc_ocl'], [1], model_name=name_model2, showfliers=True) 
    plt.xticks([0, 1], [name_model1, name_model2])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vsInc (OCL) (%)')
    plt.title('weighted vsInc (OCL) across multiple datasets')
    plt.show()
    
    # Boxplot of weighted vsInc (OCL) at the valuation date
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results_model1['weighted_vsInc_ocl_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['weighted_vsInc_ocl_val'], [1], model_name=name_model2, showfliers=True)    
    plt.xticks([0, 1], [name_model1, name_model2])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.ylabel('weighted vsInc (OCL) (%)')
    plt.title('$vsCE_{OCL} (\%)$')
    plt.show()

    # Boxplots of OCL errors at valuation date with incurreds
    bp_preds_model1 = box_plot(results_model1['ocl_error_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['ocl_error_preds_val'], [1], model_name=name_model2, showfliers=True)
    bp_incurreds = box_plot(results_model1['ocl_error_incurreds_val'], [2], model_name='Case Estimates', showfliers=True)
    plt.axhline(0, color='black', linestyle='dashed', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    #plt.ylabel('OCL error (%)')
    plt.title('$OCLerr (\%)$')
    plt.show()

    ### With Incurreds 
    # Boxplots of OCL errors at valuation date (excl. outliers)
    bp_preds_model1 = box_plot(results_model1['ocl_error_preds_val'], [0], model_name=name_model1, showfliers=False)
    bp_preds_model2 = box_plot(results_model2['ocl_error_preds_val'], [1], model_name=name_model2, showfliers=False)
    bp_incurreds = box_plot(results_model1['ocl_error_incurreds_val'], [2], model_name='Case Estimates', showfliers=False)
    plt.axhline(0, color='black', linestyle='dashed', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    #plt.ylabel('OCL error (%)')
    plt.title('$OCLerr (\%)$ (excl. outliers)')
    plt.show()

    # Boxplots of MALE and MSLE
    bp_preds_model1 = box_plot(results_model1['MALE_preds'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MALE_preds'], [1], model_name=name_model2, showfliers=True)
    bp_incurreds = box_plot(results_model1['MALE_incurreds'], [2], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    plt.title('MALE across multiple datasets')
    plt.ylabel('MALE')
    plt.show()

    bp_preds_model1 = box_plot(results_model1['MSLE_preds'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MSLE_preds'], [1], model_name=name_model2, showfliers=True)
    bp_incurreds = box_plot(results_model1['MSLE_incurreds'], [2], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    plt.title('MSLE across multiple datasets')
    plt.ylabel('MSLE')
    plt.show()

    # Boxplots of MALE and MSLE at valuation date
    bp_preds_model1 = box_plot(results_model1['MALE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MALE_preds_val'], [1], model_name=name_model2, showfliers=True)
    bp_incurreds = box_plot(results_model1['MALE_incurreds_val'], [2], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    plt.title('MALE')
    #plt.ylabel('MALE')
    plt.show()

    bp_preds_model1 = box_plot(results_model1['MSLE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MSLE_preds_val'], [1], model_name=name_model2, showfliers=True)
    bp_incurreds = box_plot(results_model1['MSLE_incurreds_val'], [2], model_name='Case Estimates', showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1, 2], [name_model1, name_model2, 'Case Estimates'])
    plt.title('MSLE')
    #plt.ylabel('MSLE')
    plt.show()

    ### Without Incurreds
    # Boxplots of OCL errors at valuation date
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results_model1['ocl_error_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['ocl_error_preds_val'], [1], model_name=name_model2, showfliers=True)
    plt.axhline(0, color='black', linestyle='dashed', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], [name_model1, name_model2])
    #plt.ylabel('OCL error (%)')
    plt.title('$OCLerr (\%)$')
    plt.show()

    # Boxplots of OCL errors at valuation date (excl. outliers)
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results_model1['ocl_error_preds_val'], [0], model_name=name_model1, showfliers=False)
    bp_preds_model2 = box_plot(results_model2['ocl_error_preds_val'], [1], model_name=name_model2, showfliers=False)
    plt.axhline(0, color='black', linestyle='dashed', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], [name_model1, name_model2])
    #plt.ylabel('OCL error (%)')
    plt.title('$OCLerr (\%)$ (excl. outliers)')
    plt.show()

    # Boxplots of MALE and MSLE at valuation date
    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results_model1['MALE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MALE_preds_val'], [1], model_name=name_model2, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], [name_model1, name_model2])
    plt.title('MALE')
    #plt.ylabel('MALE')
    plt.show()

    fig = plt.figure(figsize=(2.4, 6.4))
    bp_preds_model1 = box_plot(results_model1['MSLE_preds_val'], [0], model_name=name_model1, showfliers=True)
    bp_preds_model2 = box_plot(results_model2['MSLE_preds_val'], [1], model_name=name_model2, showfliers=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], [name_model1, name_model2])
    plt.title('MSLE')
    #plt.ylabel('MSLE')
    plt.show()

    # printing summary statistics
    print(f'\n{name_model1} OCL error: mean = {results_model1["ocl_error_preds_val"].mean():.2f}%, std = {results_model1["ocl_error_preds_val"].std():.2f}%')
    print(f'{name_model2} OCL error: mean = {results_model2["ocl_error_preds_val"].mean():.2f}%, std = {results_model2["ocl_error_preds_val"].std():.2f}%')
    print(f'Incurreds OCL error: mean = {results_model1["ocl_error_incurreds_val"].mean():.2f}%, std = {results_model1["ocl_error_incurreds_val"].std():.2f}%')

    print(f'\n{name_model1} MALE: mean = {results_model1["MALE_preds_val"].mean():.4f}, std = {results_model1["MALE_preds_val"].std():.4f}')
    print(f'{name_model2} MALE: mean = {results_model2["MALE_preds_val"].mean():.4f}, std = {results_model2["MALE_preds_val"].std():.4f}')
    print(f'Incurreds MALE: mean = {results_model1["MALE_incurreds_val"].mean():.4f}, std = {results_model1["MALE_incurreds_val"].std():.4f}')

    print(f'\n{name_model1} MSLE: mean = {results_model1["MSLE_preds_val"].mean():.4f}, std = {results_model1["MSLE_preds_val"].std():.4f}')
    print(f'{name_model2} MSLE: mean = {results_model2["MSLE_preds_val"].mean():.4f}, std = {results_model2["MSLE_preds_val"].std():.4f}')
    print(f'Incurreds MSLE: mean = {results_model1["MSLE_incurreds_val"].mean():.4f}, std = {results_model1["MSLE_incurreds_val"].std():.4f}')

    print(f'\n{name_model1} weighted vsInc (OCL): mean = {results_model1["weighted_vsInc_ocl_val"].mean():.2f}%, std = {results_model1["weighted_vsInc_ocl_val"].std():.2f}%')
    print(f'{name_model2} weighted vsInc (OCL): mean = {results_model2["weighted_vsInc_ocl_val"].mean():.2f}%, std = {results_model2["weighted_vsInc_ocl_val"].std():.2f}%')

    # generating vsM1 statistics
    vsM1_val = np.array([get_weighted_vsInc_ocl(results_model1['actuals_val'][i],
                                                results_model2['preds_val'][i],
                                                results_model1['preds_val'][i],
                                                results_model1['ocls_val'][i])
                                        for i in range(len(results_model1['actuals_val']))])
    
    print(f'vsM1 (OCL): mean = {vsM1_val.mean():.2f}%, std = {vsM1_val.std():.2f}%')

    # Boxplot of vsM1 at the valuation date
    bp_preds_model1 = box_plot(vsM1_val, [0], model_name=name_model1, showfliers=True)
    plt.xticks([0], [name_model2])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vs' + name_model1 + ' (OCL) (%) at valuation date')
    plt.title('weighted vs' + name_model1 + ' (OCL) (%) at valuation date')
    plt.show()




def plot_claim(preds_list, data, claim_no):
    '''Plots ultimate claim cost, incurred estimates and model predictions 
    for a given claim. Prelimiary function, not used in final analysis

    Inputs:
    model: the model to use
    data: a ClaimsDataset object, NOT a dataloader object
    claim_no: the claim to plot

    '''

    # Filter data for claim_no
    claim_data = data.set[data.set['claim_no']==claim_no]
    claim_index = data.index[data.index['index'].isin(claim_data['index'])]

    # Get the ultimate claim size
    ultimate = claim_index['claim_size'].values[0]

    # Get the times and incurred values
    latest_index_data = claim_data.loc[claim_data['index'] == 
                                       claim_index['index'].max(), 
                                       ['cal_time', 'paid', 'ocl']]
    
    latest_index_data['incurred'] = (latest_index_data['paid'] + 
                                     latest_index_data['ocl'])


    rnn_preds = preds_list[claim_index['index'].min():claim_index['index'].max()+1]
    rnn_pred_times = claim_data['pred_time'].unique()

    # ensuring plots finish at the same calendar time
    # RNN prediction is always later than latest incurred estimate due to 
    # the censoring method, so need to add another time to the incurred and 
    # ultimate vectors
    latest_index_data=pd.concat([latest_index_data, 
                                 pd.DataFrame({'cal_time': 
                                               rnn_pred_times[-1], 
                                               'incurred': 
                                               latest_index_data['incurred'].values[-1]}, 
                                               index = [0])], ignore_index=True)

    # plotting results
    plt.step(latest_index_data['cal_time'], 
             np.repeat(ultimate, latest_index_data.shape[0]), 
             label='ultimate', where = 'post')
    
    plt.step(latest_index_data['cal_time'], 
             latest_index_data['incurred'], 
             label='case estimate', where = 'post')
    
    plt.step(rnn_pred_times, rnn_preds, label='RNN_preds', where = 'post')
    plt.legend()

