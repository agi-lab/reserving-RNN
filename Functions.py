### IMPORTS ###################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from copy import deepcopy
from itertools import product

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
        if torch.is_tensor(preds):
            return torch.mean(torch.abs(torch.log(preds) - torch.log(actuals)))
        else:
            return np.mean(np.abs(np.log(preds) - np.log(actuals)))
    
class MeanSquaredLogError(nn.Module):
    def __init__(self):
        super(MeanSquaredLogError, self).__init__()

    def forward(self, preds, actuals):
        if torch.is_tensor(preds):
            return torch.mean(torch.square(torch.log(preds) - torch.log(actuals)))
        else:
            return np.mean(np.square(np.log(preds) - np.log(actuals)))
    
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
            
            # Special handling for LSTM forget gate bias
            if 'lstm' in name.lower() or 'gru' in name.lower():
                hidden_size = param.shape[0] // 4  # Divide into gates for LSTM
                param.data[hidden_size:2 * hidden_size] = 1.0  # Forget gate bias

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

def box_plot(data, positions, median_colour, edge_colour, fill_colour):
    data = pd.DataFrame(data)

    # dataframe boxplot needed over matplotlib or seaborn to handle missing values correctly !!!
    bp = data.boxplot(backend='matplotlib', return_type='dict', grid=False, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    
    plt.setp(bp['medians'], color=median_colour, linewidth=2)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        plt.setp(bp[element], color=edge_colour)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_colour)       
        
    return bp

### MODEL CLASSES #############################################################

class ClaimsDataset(Dataset):
    """ Based on Arkie's ClaimsDataset
    
    Notes: 
    - dataloader has to iterate from 0:len(dataset)
    - all sequences are padded to a minimum length of 50
    """

    def __init__(self, target_col, index_path, set_path, include_incurreds=True, 
                 include_covariates=False, transform_inputs=False, model_type='RNN'):
        self.target_col = target_col # string referring to name of the target column (i.e. 'claim_size', 'log_claim_size', 'log_m', 'true_ocl' or 'log_true_ocl')
        self.index = pd.read_csv(index_path) 
        self.set = pd.read_csv(set_path)
        self.include_incurreds = include_incurreds # boolean whether to use case estimate data or not
        self.include_covariates = include_covariates # boolean whether to include covariate data or not
        self.transform_inputs = transform_inputs # boolean whether to transform inputs or not
        self.model_type = model_type # string referring to the type of model being used (either 'RNN' (includes LSTM and GRU) or 'FNN')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        # Retrieves time series data, as well as summary info
        # index runs from [0, __len__(self)]
        # real_index instead refers to indexes in the csv file

        real_index = self.index['index'][index]

        df = self.set[(self.set['index']==real_index)]

        claim_no = df['claim_no'].mean()
        pred_time = df['pred_time'].mean()

        # Get relevant info from index.csv file
        target = self.index[self.target_col][index]
        claim_size = self.index['claim_size'][index]
        latest_incurred = self.index['latest_incurred'][index]
        true_ocl = self.index['true_ocl'][index]
        dev_quarter = self.index['dev_quarter'][index]
        acc_quarter = self.index['acc_quarter'][index]

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

                if self.transform_inputs:
                    databox['dev_time'] = np.log(databox['dev_time'] + 1)
                    databox['paid'] = np.log(databox['paid'] + 1)
                    databox['ocl'] = np.log(databox['ocl'] + 1)
                    
            else:
                databox = df[['dev_time','cal_time','paid']].copy()

                if self.transform_inputs:
                    databox['dev_time'] = np.log(databox['dev_time'] + 1)
                    databox['paid'] = np.log(databox['paid'] + 1)

            databox = torch.tensor(databox.values)

            # Return padded data
            if self.include_covariates:
                return (F.pad(databox.float(), (0,0,0,50-nrows)), 
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, pred_time, acc_quarter, nrows, legal_rep, injury_severity, claimant_age)

            else:
                return (F.pad(databox.float(), (0,0,0,50-nrows)), 
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no, pred_time, acc_quarter, nrows)

        # Setting up the data to be input into an FNN model
        elif self.model_type == 'FNN':
            df_copy = df.copy()

            # finding payment summary info
            df_copy['payment'] = df_copy['paid'].diff()
            df_copy['payment'] = df_copy['payment'].fillna(0)
            payment_rows = df_copy.loc[df_copy['payment'] != 0]

            num_payments = payment_rows.shape[0]
            mean_payments = payment_rows['payment'].mean() if num_payments > 0 else 0
            vco_payments = payment_rows['payment'].std() / mean_payments if num_payments > 1 else 0 # need minimum 2 payments for std dev
            max_payment = payment_rows['payment'].max() if num_payments > 0 else 0
            
            # finding case estimate summary info
            if self.include_incurreds:
                df_copy['revision'] = df_copy['ocl'].diff() + df_copy['paid'].diff()
                df_copy['revision'] = df_copy['revision'].fillna(0)
                revision_rows = df_copy.loc[df_copy['revision'] != 0]

                num_revisions = revision_rows.shape[0]
                max_revision = revision_rows['revision'].abs().max() if num_revisions > 0 else 0
                total_revisions = revision_rows['revision'].abs().sum() if num_revisions > 0 else 0
                prop_upward_revisions = (revision_rows['revision'] > 0).sum() / num_revisions if num_revisions > 0 else 0

            #print(f'df_copy:\n{df_copy}\n')
            
            if self.include_incurreds and self.include_covariates:

                '''print(f'pred_time: {pred_time}, \
                      dev_quarter: {dev_quarter}, \
                        num_payments: {num_payments}, \
                        mean_payments: {mean_payments}, \
                        vco_payments: {vco_payments}, \
                        max_payment: {max_payment}, \
                        num_revisions: {num_revisions}, \
                        max_revision: {max_revision}, \
                        total_revisions: {total_revisions}, \
                        prop_upward_revisions: {prop_upward_revisions}, \
                        legal_rep: {legal_rep}, \
                        injury_severity: {injury_severity}, \
                        claimant_age: {claimant_age}, \
                        target: {target}, \
                        claim_size: {claim_size}, \
                        latest_incurred: {latest_incurred}, \
                        true_ocl: {true_ocl}, \
                        real_index: {real_index}, \
                        claim_no: {claim_no}')'''

                return (pred_time, dev_quarter, acc_quarter, num_payments, mean_payments, vco_payments, max_payment, 
                        num_revisions, max_revision, total_revisions, prop_upward_revisions, 
                        legal_rep, injury_severity, claimant_age,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no)
            
            elif self.include_incurreds and not self.include_covariates:

                '''print(f'pred_time: {pred_time}, \
                      dev_quarter: {dev_quarter}, \
                        num_payments: {num_payments}, \
                        mean_payments: {mean_payments}, \
                        vco_payments: {vco_payments}, \
                        max_payment: {max_payment}, \
                        num_revisions: {num_revisions}, \
                        max_revision: {max_revision}, \
                        total_revisions: {total_revisions}, \
                        prop_upward_revisions: {prop_upward_revisions}, \
                        target: {target}, \
                        claim_size: {claim_size}, \
                        latest_incurred: {latest_incurred}, \
                        true_ocl: {true_ocl}, \
                        real_index: {real_index}, \
                        claim_no: {claim_no}')'''

                return (pred_time, dev_quarter, acc_quarter, num_payments, mean_payments, vco_payments, max_payment, 
                        num_revisions, max_revision, total_revisions, prop_upward_revisions,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no)
            
            elif not self.include_incurreds and self.include_covariates:

                '''print(f'pred_time: {pred_time}, \
                      dev_quarter: {dev_quarter}, \
                        num_payments: {num_payments}, \
                        mean_payments: {mean_payments}, \
                        vco_payments: {vco_payments}, \
                        max_payment: {max_payment}, \
                        legal_rep: {legal_rep}, \
                        injury_severity: {injury_severity}, \
                        claimant_age: {claimant_age}, \
                        target: {target}, \
                        claim_size: {claim_size}, \
                        latest_incurred: {latest_incurred}, \
                        true_ocl: {true_ocl}, \
                        real_index: {real_index}, \
                        claim_no: {claim_no}')'''

                return (pred_time, dev_quarter, acc_quarter, num_payments, mean_payments, vco_payments, max_payment,
                        legal_rep, injury_severity, claimant_age,
                        target, claim_size, latest_incurred, true_ocl, real_index, 
                        claim_no)

            else:

                '''print(f'pred_time: {pred_time}, \
                      dev_quarter: {dev_quarter}, \
                        num_payments: {num_payments}, \
                        mean_payments: {mean_payments}, \
                        vco_payments: {vco_payments}, \
                        max_payment: {max_payment}, \
                        target: {target}, \
                        claim_size: {claim_size}, \
                        latest_incurred: {latest_incurred}, \
                        true_ocl: {true_ocl}, \
                        real_index: {real_index}, \
                        claim_no: {claim_no}')'''

                return (pred_time, dev_quarter, acc_quarter, num_payments, mean_payments, vco_payments, max_payment,
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
        self.nConcatUnits = 64 # hard coding 64 units for now, this will be the number of units of both RNN and static inputs before concatenating


        # nFeatures is the number of features to be input into the RNN layer
        self.nFeatures = 3 + self.include_incurreds # 4 features with ocl, 3 without

        if self.normalisation:
            self.layer_norm1 = nn.LayerNorm(self.nFeatures)
            self.dropout_layer = nn.Dropout(self.dropout)

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

            self.batch_norm1 = nn.BatchNorm1d(self.nConcatUnits)
            self.batch_norm2 = nn.BatchNorm1d(self.nConcatUnits)
            self.batch_norm3 = nn.BatchNorm1d(self.nConcatUnits)

        else:
            if type == 'RNN':
                self.rnn = nn.RNN(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, nonlinearity=nonlinearity, 
                                dropout=dropout)

            elif type == 'LSTM':
                self.rnn = nn.LSTM(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, dropout=dropout)
            
            elif type == 'GRU':
                self.rnn = nn.GRU(self.nFeatures, nHidden, nLayers, 
                                batch_first=True, dropout=dropout)
            
            else:
                raise ValueError("type must be 'RNN', 'LSTM' or 'GRU'")

        # RNN output is reduced in size
        self.fc1 = nn.Linear(nHidden, self.nConcatUnits) # hard coding 64 units for now


        if self.include_covariates:
            self.embedding_dim = 2
            self.embedding_sev = nn.Embedding(6, self.embedding_dim) # 6 possible injury severities, output 2 dimensions
            self.embedding_age = nn.Embedding(5, self.embedding_dim) # 5 possible ages, output 2 dimensions

            # static inputs are increased in size
            self.fc2 = nn.Linear(3 + 2 * self.embedding_dim, self.nConcatUnits)

            # combining RNN and static outputs
            self.fc3 = nn.Linear(2 * self.nConcatUnits, self.nConcatUnits)

        else:
            # otherwise RNN outputs + 2 for pred time and accident quarter
            self.fc3 = nn.Linear(self.nConcatUnits + 2, self.nConcatUnits)

        self.fc4 = nn.Linear(self.nConcatUnits, nOut)

    def forward(self, x):
        # x[0] will be the packed datapoints, x[1:] will be the static covariates
        if self.normalisation:
            out, nrows = pad_packed_sequence(x[0], batch_first=True)
            out = self.layer_norm1(out)
            out = pack_padded_sequence(out, nrows, batch_first=True, enforce_sorted=False)

            for i, rnn in enumerate(self.rnn_layers):
                out, ht = rnn(out)  # RNN output

                out, nrows = pad_packed_sequence(out, batch_first=True)
                out = self.layer_norms_rnn[i](out)
                if i < self.nLayers - 1:
                    out = self.dropout_layer(out)
                out = pack_padded_sequence(out, nrows, batch_first=True, enforce_sorted=False)
        
        else:
            out, ht = self.rnn(x[0])

        if self.type == 'LSTM':
            ht = ht[0]

        if self.normalisation:
            ht = self.layer_norm2(ht)

        out = self.fc1(ht[-1,:,:])
        out = self.relu(out)

        if self.normalisation:
            out = self.batch_norm1(out)

        if self.include_covariates:
            sev_embed = self.embedding_sev(x[4].long())
            age_embed = self.embedding_age(x[5].long())

            static_out = torch.cat((x[1], x[2], x[3], sev_embed[:, -1, :], age_embed[:, -1, :]), 1)
            static_out = self.fc2(static_out)
            static_out = self.relu(static_out)

            if self.normalisation:
                static_out = self.batch_norm2(static_out)

            out = torch.cat((out, static_out), 1)
            
        else:
            out = torch.cat((out, x[1], x[2]), 1)

        out = self.fc3(out)
        out = self.relu(out)

        if self.normalisation:
            out = self.batch_norm3(out)      

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

        # 3 variables for transaction times, 4 for payments, 5 for revisions, 3 for covariates
        self.num_features = 7 + 5 * include_incurreds + 3 * include_covariates
        self.final_activation = final_activation
        self.normalisation = normalisation

        if self.normalisation:
            layers = [nn.BatchNorm1d(self.num_features), nn.Linear(self.num_features, nHidden), nn.ReLU(), nn.Dropout(dropout)]

            for _ in range(nLayers - 1):
                layers.append(nn.BatchNorm1d(nHidden))
                layers.append(nn.Linear(nHidden, nHidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.BatchNorm1d(nHidden))

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
        if self.final_activation == 'exponential':
            out = torch.exp(self.nn_output_layer(x).squeeze(-1)) # * x[:, -1].squeeze(-1)
        elif self.final_activation == 'softplus':
            out = F.softplus(self.nn_output_layer(x).squeeze(-1))
        elif self.final_activation == 'linear':
            out = self.nn_output_layer(x).squeeze(-1)
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
        val_loss_list = []
        val_vsInc_list = []
        val_weighted_vsInc_claimsize_list = []
        val_weighted_vsInc_ocl_list = []
        val_uie_list = []
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
                                              shuffle=True, drop_last=True,
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
        total_loss = 0
        total_datapoints = 0
        total_vsInc = 0
        total_weighted_vsInc_claimsize = 0
        total_weighted_vsInc_ocl = 0
        total_uie = 0
        total_ultimates = 0
        total_ocls = 0

        for batch in trainloader:

            if hp_comb['model_type'] == 'RNN':
                # extract batch data
                if hp_comb['include_covariates']:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls,
                    indexes, claim_nos, pred_times, acc_quarters, nrowss, legal_reps, 
                    injury_severities, claimant_ages) = batch

                    legal_reps = legal_reps.unsqueeze(1).to(device).float()
                    injury_severities = injury_severities.unsqueeze(1).to(device).float()
                    claimant_ages = claimant_ages.unsqueeze(1).to(device).float()
                    
                else:
                    (datapoints, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos, pred_times, acc_quarters, nrowss) = batch
                    
                datapoints = datapoints.to(device).float()
                targets = targets.to(device).float()
                claim_sizes = claim_sizes.to(device).float()
                latest_incurreds = latest_incurreds.to(device).float()
                true_ocls = true_ocls.to(device).float()
                pred_times = pred_times.unsqueeze(1).to(device).float()
                acc_quarters = acc_quarters.unsqueeze(1).to(device).float()

                packed = pack_padded_sequence(datapoints, nrowss, 
                                            enforce_sorted=False, 
                                            batch_first=True)

                if hp_comb['include_covariates']:
                    packed_extra = (packed, pred_times, acc_quarters, legal_reps, 
                                    injury_severities, claimant_ages)

                else:
                    packed_extra = (packed, pred_times, acc_quarters)  

            elif hp_comb['model_type'] == 'FNN':
                if hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, num_revisions, max_revision, 
                    total_revisions, prop_upward_revisions, legal_reps, 
                    injury_severities, claimant_ages, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    num_revisions = num_revisions.to(device).float()
                    max_revision = max_revision.to(device).float()
                    total_revisions = total_revisions.to(device).float()
                    prop_upward_revisions = prop_upward_revisions.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    #print(f'type of pred_times: {type(pred_times)}')
                    #print(f'dimension of pred_times: {pred_times.shape}')

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, latest_incurreds, num_revisions, max_revision,
                                    total_revisions, prop_upward_revisions, legal_reps,
                                    injury_severities, claimant_ages), dim=1).to(device)

                    #print(f'dimension of packed_extra: {packed_extra.shape}')

                elif hp_comb['include_incurreds'] and not hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, num_revisions, max_revision, 
                    total_revisions, prop_upward_revisions, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    num_revisions = num_revisions.to(device).float()
                    max_revision = max_revision.to(device).float()
                    total_revisions = total_revisions.to(device).float()
                    prop_upward_revisions = prop_upward_revisions.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, latest_incurreds, num_revisions, max_revision,
                                    total_revisions, prop_upward_revisions), dim=1).to(device)

                elif not hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, legal_reps, injury_severities, 
                    claimant_ages, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, legal_reps, injury_severities, 
                                    claimant_ages), dim=1).to(device)

                else:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, 
                                                num_payments, mean_payments,
                                                vco_payments, max_payment), dim=1).to(device)

            else:
                raise ValueError("model_type must be 'RNN' or 'FNN'")

            raw_preds = model(packed_extra)    
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            # converting raw preds and targets to be in terms of ultimate claim size
            if train_data.target_col == 'claim_size':
                preds = raw_preds
                ultimates = targets
            
            elif train_data.target_col == 'log_claim_size':
                preds = torch.exp(raw_preds)
                ultimates = torch.exp(targets)
            
            elif train_data.target_col == 'log_m':
                preds = torch.exp(raw_preds) * latest_incurreds
                ultimates = torch.exp(targets) * latest_incurreds
            
            elif train_data.target_col == 'true_ocl':
                preds = raw_preds + claim_sizes - true_ocls
                ultimates = targets + claim_sizes - true_ocls

            elif train_data.target_col == 'log_true_ocl':
                preds = torch.exp(raw_preds) + claim_sizes - true_ocls
                ultimates = torch.exp(targets) + claim_sizes - true_ocls

            else:
                ValueError('Invalid target, must be "claim_size", "log_claim_size", "log_m", "true_ocl" or "log_true_ocl"')


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
            total_loss += loss.item() * preds.size(0)
            total_datapoints += preds.size(0)
            total_vsInc += sum(torch.abs((ultimates-preds)) < 
                               torch.abs((ultimates-latest_incurreds)))
            
            total_weighted_vsInc_claimsize += sum(ultimates * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_weighted_vsInc_ocl += sum(true_ocls * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_ultimates += sum(ultimates)
            total_ocls += sum(true_ocls)
            
            # uie is not being included in the paper, but useful as a diagnostic during training
            total_uie+=sum(torch.logical_and((preds < latest_incurreds), 
                                             (torch.abs((ultimates-preds)) > 
                                              torch.abs((ultimates-
                                                         latest_incurreds)))))

        # End of epoch summary
        vs_incurred_accuracy = total_vsInc / total_datapoints * 100
        uie = total_uie / total_datapoints * 100
        total_loss = total_loss / total_datapoints
        weighted_vsinc_claimsize = total_weighted_vsInc_claimsize / total_ultimates * 100
        weighted_vsinc_ocl = total_weighted_vsInc_ocl / total_ocls * 100

        if verbose:
            print(f'Epoch {epoch}: '
                  f'training loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc (Claim Size) = {weighted_vsinc_claimsize:.2f}%, '
                  f'weighted vsInc (OCL) = {weighted_vsinc_ocl:.2f}%, '
                  f'UIE = {uie:.2f}%')

        # Validation
        if val_data:
            
            if verbose:
                print('Validation')
                
            test_network(model, val_data, hp_comb, val_loss_list=val_loss_list, 
                         val_vsInc_list=val_vsInc_list, 
                         val_weighted_vsInc_claimsize_list=val_weighted_vsInc_claimsize_list,
                         val_weighted_vsInc_ocl_list=val_weighted_vsInc_ocl_list,
                         val_uie_list=val_uie_list, verbose=verbose)

            # Early stopping
            min_delta = 0.0001

            if val_loss_list[-1] < best_val_loss - min_delta:
                best_val_loss = val_loss_list[-1]
                best_val_vsInc = val_vsInc_list[-1].item()
                best_val_weighted_vsInc_claimsize = val_weighted_vsInc_claimsize_list[-1].item()
                best_val_weighted_vsInc_ocl = val_weighted_vsInc_ocl_list[-1].item()
                best_val_uie = val_uie_list[-1].item()
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
            
def test_network(model, test_data, hp_comb, preds_list=None, verbose=True, 
                 val_loss_list=None, val_vsInc_list=None, 
                 val_weighted_vsInc_claimsize_list=None, 
                 val_weighted_vsInc_ocl_list=None, val_uie_list=None):
    
    """Args:
        preds_list: empty list to append predictions to
        val_loss_list/val_vsInc_list/val_uie_list: lists to be passed to keep 
        track of within-model stats during training
    """

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                              batch_size=hp_comb['batch_size'], 
                                              shuffle=False,
                                              num_workers=4, pin_memory=True)

    total_loss = 0
    total_datapoints = 0
    total_vsInc = 0
    total_weighted_vsInc_claimsize = 0
    total_weighted_vsInc_ocl = 0
    total_uie = 0
    total_ultimates = 0
    total_ocls = 0

    # set model to test mode
    model.eval()

    # Test the model
    with torch.no_grad():
        for batch in test_loader:

            if hp_comb['model_type'] == 'RNN':

                if hp_comb['include_covariates']:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                    indexes, claim_nos, pred_times, acc_quarters, nrowss, legal_reps, 
                    injury_severities, claimant_ages) = batch

                    legal_reps = legal_reps.unsqueeze(1).to(device).float()
                    injury_severities = injury_severities.unsqueeze(1).to(device).float()
                    claimant_ages = claimant_ages.unsqueeze(1).to(device).float()

                else:
                    (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                    indexes, claim_nos, pred_times, acc_quarters, nrowss) = batch

                datapoints = datapoints.to(device).float()
                targets = targets.to(device).float()
                claim_sizes = claim_sizes.to(device).float()
                latest_incurreds = latest_incurreds.to(device).float()
                true_ocls = true_ocls.to(device).float()
                pred_times = pred_times.unsqueeze(1).to(device).float()
                acc_quarters = acc_quarters.unsqueeze(1).to(device).float()

                packed = pack_padded_sequence(datapoints, nrowss, 
                                            enforce_sorted=False, 
                                            batch_first=True)

                if hp_comb['include_covariates']:
                    packed_extra = (packed, pred_times, acc_quarters, legal_reps, 
                                    injury_severities, claimant_ages)
                    
                else:
                    packed_extra = (packed, pred_times, acc_quarters)

            elif hp_comb['model_type'] == 'FNN':
                if hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, num_revisions, max_revision, 
                    total_revisions, prop_upward_revisions, legal_reps, 
                    injury_severities, claimant_ages, targets, claim_sizes, 
                    latest_incurreds, true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    num_revisions = num_revisions.to(device).float()
                    max_revision = max_revision.to(device).float()
                    total_revisions = total_revisions.to(device).float()
                    prop_upward_revisions = prop_upward_revisions.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, latest_incurreds, num_revisions, max_revision,
                                    total_revisions, prop_upward_revisions, legal_reps,
                                    injury_severities, claimant_ages), dim=1).to(device)

                elif hp_comb['include_incurreds'] and not hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, num_revisions, max_revision, 
                    total_revisions, prop_upward_revisions, targets, claim_sizes,
                    latest_incurreds, true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    num_revisions = num_revisions.to(device).float()
                    max_revision = max_revision.to(device).float()
                    total_revisions = total_revisions.to(device).float()
                    prop_upward_revisions = prop_upward_revisions.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, latest_incurreds, num_revisions, max_revision,
                                    total_revisions, prop_upward_revisions), dim=1).to(device)

                elif not hp_comb['include_incurreds'] and hp_comb['include_covariates']:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, legal_reps, injury_severities, 
                    claimant_ages, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    legal_reps = legal_reps.to(device).float()
                    injury_severities = injury_severities.to(device).float()
                    claimant_ages = claimant_ages.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, num_payments, mean_payments,
                                    vco_payments, max_payment, legal_reps, injury_severities, 
                                    claimant_ages), dim=1).to(device)
                
                else:
                    (pred_times, dev_quarters, acc_quarters, num_payments, mean_payments, 
                    vco_payments, max_payment, targets, claim_sizes, latest_incurreds, 
                    true_ocls, indexes, claim_nos) = batch

                    pred_times = pred_times.to(device).float()
                    dev_quarters = dev_quarters.to(device).float()
                    acc_quarters = acc_quarters.to(device).float()
                    num_payments = num_payments.to(device).float()
                    mean_payments = mean_payments.to(device).float()
                    vco_payments = vco_payments.to(device).float()
                    max_payment = max_payment.to(device).float()
                    targets = targets.to(device).float()
                    claim_sizes = claim_sizes.to(device).float()
                    latest_incurreds = latest_incurreds.to(device).float()
                    true_ocls = true_ocls.to(device).float()

                    packed_extra = torch.stack((pred_times, dev_quarters, acc_quarters, 
                                                num_payments, mean_payments,
                                                vco_payments, max_payment), dim=1).to(device)

            else:
                raise ValueError("model_type must be 'RNN' or 'FNN'")

            raw_preds = model(packed_extra)
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            if test_data.target_col == 'claim_size':
                preds = raw_preds
                ultimates = targets

            elif test_data.target_col == 'log_claim_size':
                preds = torch.exp(raw_preds)
                ultimates = torch.exp(targets)
            
            elif test_data.target_col == 'log_m':
                preds = torch.exp(raw_preds) * latest_incurreds
                ultimates = torch.exp(targets) * latest_incurreds

            elif test_data.target_col == 'true_ocl':
                preds = raw_preds + claim_sizes - true_ocls
                ultimates = targets + claim_sizes - true_ocls

            elif test_data.target_col == 'log_true_ocl':
                preds = torch.exp(raw_preds) + claim_sizes - true_ocls
                ultimates = torch.exp(targets) + claim_sizes - true_ocls

            else:
                ValueError('Invalid target, must be "claim_size", "log_claim_size", "log_m", "true_ocl" or "log_true_ocl"')


            # Loss and gradient descent
            if isinstance(hp_comb['criterion'], MSLE_with_penalty):
                loss = hp_comb['criterion'](raw_preds, targets, claim_sizes - true_ocls, preds)

            else:
                loss = hp_comb['criterion'](raw_preds, targets)

            # Track statistics
            total_loss += loss.item() * preds.size(0)
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

            if preds_list is not None:
                preds_list.extend([pred.item() for pred in preds])

        # End of epoch summary
        vs_incurred_accuracy = total_vsInc / total_datapoints * 100
        uie = total_uie / total_datapoints * 100
        total_loss = total_loss / total_datapoints
        weighted_vsinc_claimsize = total_weighted_vsInc_claimsize / total_ultimates * 100
        weighted_vsinc_ocl = total_weighted_vsInc_ocl / total_ocls * 100

        if verbose:
            print(f'loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc (Claim Size) = {weighted_vsinc_claimsize:.2f}%, '
                  f'weighted vsInc (OCL) = {weighted_vsinc_ocl:.2f}%, '
                  f'UIE = {uie:.2f}%')

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
                 val_weighted_vsInc_ocl_list=None, val_uie_list=None)

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

def aggregate_by_time(index_data, actuals, preds, incurreds, ocls, time_str):
    '''Plots aggregate claims, vsInc and weighted vsInc over time, 
       either calendar or development
       
       time_str should be either 'pred_time' (for calendar quarter results), 
       'dev_quarter' for 'development' quarter results (i.e. time since notification, not since occurrence),
       'acc_quarter' for accident quarter results,
       or 'rept_quarter' for reported quarter results'''

    # 1 dataset, 1 prediction
    if isinstance(preds, pd.Series):
        times = np.sort(index_data[time_str].unique())

        actuals_by_time = np.zeros(len(times))
        incurreds_by_time = np.zeros(len(times))
        ocls_by_time = np.zeros(len(times))

        preds_by_time = np.zeros(len(times))
        vsInc_by_time = np.zeros(len(times))
        weighted_vsInc_claimsize_by_time = np.zeros(len(times))
        weighted_vsInc_ocl_by_time = np.zeros(len(times))

        pred_count_by_time = np.zeros(len(times))

    else:
        
        # 1 dataset, multiple predictions
        if isinstance(actuals, pd.Series):
            times = np.sort(index_data[time_str].unique())

            actuals_by_time = np.zeros(len(times))
            incurreds_by_time = np.zeros(len(times))
            ocls_by_time = np.zeros(len(times))

            pred_count_by_time = np.zeros(len(times))

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
            ocl_incurreds_over_actuals_by_time = np.zeros((len(incurreds), len(times)))

            mean_actuals_by_time = np.zeros(len(times))
            mean_incurreds_by_time = np.zeros(len(times))
            sd_incurreds_by_time = np.zeros(len(times))
            mean_ocls_by_time = np.zeros(len(times))
            sd_ocls_by_time = np.zeros(len(times))

            mean_preds_over_actuals_by_time = np.zeros(len(times))
            sd_preds_over_actuals_by_time = np.zeros(len(times))
            mean_incurreds_over_actuals_by_time = np.zeros(len(times))
            sd_incurreds_over_actuals_by_time = np.zeros(len(times))

            q1_preds_over_actuals_by_time = np.zeros(len(times))
            q3_preds_over_actuals_by_time = np.zeros(len(times))
            q1_incurreds_over_actuals_by_time = np.zeros(len(times))
            q3_incurreds_over_actuals_by_time = np.zeros(len(times))
            q2_preds_over_actuals_by_time = np.zeros(len(times))
            q2_incurreds_over_actuals_by_time = np.zeros(len(times))

            q1_weighted_vsInc_claimsize_by_time = np.zeros(len(times))
            q3_weighted_vsInc_claimsize_by_time = np.zeros(len(times))
            q2_weighted_vsInc_claimsize_by_time = np.zeros(len(times))

            q1_weighted_vsInc_ocl_by_time = np.zeros(len(times))
            q3_weighted_vsInc_ocl_by_time = np.zeros(len(times))
            q2_weighted_vsInc_ocl_by_time = np.zeros(len(times))

            pred_count_by_time = np.zeros((len(preds), len(times)))

        preds_by_time = np.zeros((len(preds), len(times)))
        mean_preds_by_time = np.zeros(len(times))
        sd_preds_by_time = np.zeros(len(times))
        vsInc_by_time = np.zeros((len(preds), len(times)))
        mean_vsInc_by_time = np.zeros(len(times))
        sd_vsInc_by_time = np.zeros(len(times))
        weighted_vsInc_claimsize_by_time = np.zeros((len(preds), len(times)))
        mean_weighted_vsInc_claimsize_by_time = np.zeros(len(times))
        sd_weighted_vsInc_claimsize_by_time = np.zeros(len(times))
        weighted_vsInc_ocl_by_time = np.zeros((len(preds), len(times)))
        mean_weighted_vsInc_ocl_by_time = np.zeros(len(times))
        sd_weighted_vsInc_ocl_by_time = np.zeros(len(times))
        

    for index, time in enumerate(times):

        # 1 dataset, 1 prediction
        if isinstance(preds, pd.Series):
            indicator = index_data[time_str] == time

            actuals_by_time[index] = np.sum(actuals[indicator])
            incurreds_by_time[index] = np.sum(incurreds[indicator])
            ocls_by_time[index] = np.sum(ocls[indicator])

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
            
            pred_count_by_time[index] = np.count_nonzero(preds[indicator])
            
        else:
            
            # 1 dataset, multiple predictions
            if isinstance(actuals, pd.Series):
                indicator = index_data[time_str] == time

                preds_by_time[:, index] = np.sum(preds[:, indicator], axis=1)
                actuals_by_time[index] = np.sum(actuals[indicator])
                incurreds_by_time[index] = np.sum(incurreds[indicator])
                ocls_by_time[index] = np.sum(ocls[indicator])

                vsInc_by_time[:, index] = np.array([get_vsInc(actuals[indicator], 
                                                    preds[i, indicator], 
                                                    incurreds[indicator]) 
                                        for i in range(preds.shape[0])])

                weighted_vsInc_claimsize_by_time[:, index] = np.array([get_weighted_vsInc_claimsize(actuals[indicator], 
                                                                    preds[i, indicator], 
                                                                    incurreds[indicator]) 
                                                for i in range(preds.shape[0])])

                weighted_vsInc_ocl_by_time[:, index] = np.array([get_weighted_vsInc_ocl(actuals[indicator], 
                                                                    preds[i, indicator], 
                                                                    incurreds[indicator], 
                                                                    ocls[indicator]) 
                                                for i in range(preds.shape[0])])
                
                pred_count_by_time[index] = np.count_nonzero(actuals[indicator])

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
                    ocl_incurreds_over_actuals_by_time[i, index] = (np.sum(incurreds[i][indicator]) - paids_by_time[i, index]) / ocls_by_time[i, index] if ocls_by_time[i, index] > 0 else 1

                    pred_count_by_time[i, index] = np.count_nonzero(preds[i][indicator])
                
                mean_actuals_by_time[index] = np.mean(actuals_by_time[:, index], axis=0)

                mean_incurreds_by_time[index] = np.mean(incurreds_by_time[:, index], axis=0)
                sd_incurreds_by_time[index] = np.std(incurreds_by_time[:, index], axis=0)

                mean_ocls_by_time[index] = np.mean(ocls_by_time[:, index], axis=0)
                sd_ocls_by_time[index] = np.std(ocls_by_time[:, index], axis=0)

                mean_preds_over_actuals_by_time[index] = np.mean(preds_over_actuals_by_time[:, index], axis=0)
                sd_preds_over_actuals_by_time[index] = np.std(preds_over_actuals_by_time[:, index], axis=0)
                mean_incurreds_over_actuals_by_time[index] = np.mean(incurreds_over_actuals_by_time[:, index], axis=0)
                sd_incurreds_over_actuals_by_time[index] = np.std(incurreds_over_actuals_by_time[:, index], axis=0)

                q1_preds_over_actuals_by_time[index] = np.percentile(preds_over_actuals_by_time[:, index], 25)
                q3_preds_over_actuals_by_time[index] = np.percentile(preds_over_actuals_by_time[:, index], 75)
                q1_incurreds_over_actuals_by_time[index] = np.percentile(incurreds_over_actuals_by_time[:, index], 25)
                q3_incurreds_over_actuals_by_time[index] = np.percentile(incurreds_over_actuals_by_time[:, index], 75)
                q2_preds_over_actuals_by_time[index] = np.percentile(preds_over_actuals_by_time[:, index], 50)
                q2_incurreds_over_actuals_by_time[index] = np.percentile(incurreds_over_actuals_by_time[:, index], 50)

                q1_weighted_vsInc_claimsize_by_time[index] = np.percentile(weighted_vsInc_claimsize_by_time[:, index], 25)
                q3_weighted_vsInc_claimsize_by_time[index] = np.percentile(weighted_vsInc_claimsize_by_time[:, index], 75)
                q2_weighted_vsInc_claimsize_by_time[index] = np.percentile(weighted_vsInc_claimsize_by_time[:, index], 50)

                q1_weighted_vsInc_ocl_by_time[index] = np.percentile(weighted_vsInc_ocl_by_time[:, index], 25)
                q3_weighted_vsInc_ocl_by_time[index] = np.percentile(weighted_vsInc_ocl_by_time[:, index], 75)
                q2_weighted_vsInc_ocl_by_time[index] = np.percentile(weighted_vsInc_ocl_by_time[:, index], 50)

            mean_preds_by_time[index] = np.mean(preds_by_time[:, index], axis=0)
            sd_preds_by_time[index] = np.std(preds_by_time[:, index], axis=0)

            mean_vsInc_by_time[index] = np.mean(vsInc_by_time[:, index], axis=0)
            sd_vsInc_by_time[index] = np.std(vsInc_by_time[:, index], axis=0)

            mean_weighted_vsInc_claimsize_by_time[index] = np.mean(weighted_vsInc_claimsize_by_time[:, index], axis=0)
            sd_weighted_vsInc_claimsize_by_time[index] = np.std(weighted_vsInc_claimsize_by_time[:, index], axis=0)

            mean_weighted_vsInc_ocl_by_time[index] = np.mean(weighted_vsInc_ocl_by_time[:, index], axis=0)
            sd_weighted_vsInc_ocl_by_time[index] = np.std(weighted_vsInc_ocl_by_time[:, index], axis=0)

    # Converting pred count to proportion
    if isinstance(preds, pd.Series) or isinstance(actuals, pd.Series):
        # 1 dataset
        pred_cumulative_count_by_time = np.cumsum(pred_count_by_time)
        pred_cumulative_prop_by_time = pred_cumulative_count_by_time / pred_cumulative_count_by_time[-1]

    else:
        # multiple datasets

        # sum over multiple datasets (could change this later to have a boxplot of claim counts)
        pred_count_by_time = np.sum(pred_count_by_time, axis=0)

        # now has the same shape as for 1 dataset, so can use the same code
        pred_cumulative_count_by_time = np.cumsum(pred_count_by_time)
        pred_cumulative_prop_by_time = pred_cumulative_count_by_time / pred_cumulative_count_by_time[-1]


    # 1 dataset, 1 prediction
    if isinstance(preds, pd.Series):
        # plotting aggregate preds
        plt.plot(times, actuals_by_time / actuals_by_time, label='Actuals')
        plt.plot(times, preds_by_time / actuals_by_time, label='Predictions')
        plt.plot(times, incurreds_by_time / actuals_by_time, label='Incurreds')
        plt.legend(loc='upper right')
        plt.title('Aggregate claim sizes (as proportion of actual)')

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
        plt.plot(times, vsInc_by_time)
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
        plt.plot(times, weighted_vsInc_claimsize_by_time)
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
        plt.plot(times, weighted_vsInc_ocl_by_time)
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
            # plotting aggregate preds
            plt.plot(times, actuals_by_time / actuals_by_time, label='Actuals')
            plt.plot(times, mean_preds_by_time / actuals_by_time, label='Predictions')
            plt.fill_between(times, (mean_preds_by_time - sd_preds_by_time) / actuals_by_time, 
                            (mean_preds_by_time + sd_preds_by_time) / actuals_by_time, alpha=0.3, color='orange')
            plt.fill_between(times, (mean_preds_by_time - 2*sd_preds_by_time) / actuals_by_time, 
                            (mean_preds_by_time + 2*sd_preds_by_time) / actuals_by_time, alpha=0.2, color='orange')
            plt.plot(times, incurreds_by_time / actuals_by_time, label='Incurreds')
            plt.legend(loc='upper right')
            plt.title('Aggregate claim sizes (as proportion of actual)')

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

        # multiple datasets, multiple predictions
        else:

            # plotting aggregate preds (new version with corrected ratios)
            plt.plot(times, mean_actuals_by_time / mean_actuals_by_time, label='Actuals')
            plt.plot(times, mean_preds_over_actuals_by_time, label='Predictions')
            plt.fill_between(times, (mean_preds_over_actuals_by_time - sd_preds_over_actuals_by_time), 
                            (mean_preds_over_actuals_by_time + sd_preds_over_actuals_by_time), alpha=0.3, color='orange')
            plt.fill_between(times, (mean_preds_over_actuals_by_time - 2*sd_preds_over_actuals_by_time), 
                            (mean_preds_over_actuals_by_time + 2*sd_preds_over_actuals_by_time), alpha=0.2, color='orange')
            plt.plot(times, mean_incurreds_over_actuals_by_time, label='Incurreds')
            plt.fill_between(times, (mean_incurreds_over_actuals_by_time - sd_incurreds_over_actuals_by_time),
                            (mean_incurreds_over_actuals_by_time + sd_incurreds_over_actuals_by_time), alpha=0.3, color='green')
            plt.fill_between(times, (mean_incurreds_over_actuals_by_time - 2*sd_incurreds_over_actuals_by_time),
                            (mean_incurreds_over_actuals_by_time + 2*sd_incurreds_over_actuals_by_time), alpha=0.2, color='green')
            plt.legend(loc='upper right')
            plt.title('Aggregate claim sizes (as proportion of actual) (corrected ratios)')
            plt.grid(axis='both', linestyle='--', alpha=0.7)

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

            # Plotting aggregate preds (now with q1-q3 bands and q2 instead of mean)
            plt.plot(times, mean_actuals_by_time / mean_actuals_by_time, label='Actuals')
            plt.plot(times, q2_preds_over_actuals_by_time, label='Predictions')
            plt.fill_between(times, q1_preds_over_actuals_by_time, 
                            q3_preds_over_actuals_by_time, alpha=0.3, color='orange')
            plt.plot(times, q2_incurreds_over_actuals_by_time, label='Incurreds')
            plt.fill_between(times, q1_incurreds_over_actuals_by_time,
                            q3_incurreds_over_actuals_by_time, alpha=0.3, color='green')
            plt.legend(loc='upper right')
            plt.title('Aggregate claim sizes (as proportion of actual) (q2 with q1-q3 bands)')
            plt.grid(axis='both', linestyle='--', alpha=0.7)

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

            # Boxplot with better colours
            bp_preds = box_plot(preds_over_actuals_by_time, positions=times, median_colour='red', edge_colour='chocolate', fill_colour='bisque')
            bp_incurreds = box_plot(incurreds_over_actuals_by_time, positions=times, median_colour='cyan', edge_colour='darkgreen', fill_colour='lightgreen')
            plt.plot(times, mean_actuals_by_time / mean_actuals_by_time)
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)
            plt.legend([bp_preds["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Incurreds'])
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
            bp_preds = box_plot(ocl_preds_over_actuals_by_time, positions=times, median_colour='red', edge_colour='chocolate', fill_colour='bisque')
            bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, median_colour='cyan', edge_colour='darkgreen', fill_colour='lightgreen')
            plt.plot(times, mean_actuals_by_time / mean_actuals_by_time)
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)
            plt.legend([bp_preds["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Incurreds'])
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
            
            plt.title('Aggregate OCLs (as proportion of actual)')
            plt.ylabel('Ratio')
            plt.show()

            # Boxplot with aggregate OCLs (hard code negative OCL preds as 0)
            ocl_preds_over_actuals_by_time[ocl_preds_over_actuals_by_time < 0] = 0

            bp_preds = box_plot(ocl_preds_over_actuals_by_time, positions=times, median_colour='red', edge_colour='chocolate', fill_colour='bisque')
            bp_incurreds = box_plot(ocl_incurreds_over_actuals_by_time, positions=times, median_colour='cyan', edge_colour='darkgreen', fill_colour='lightgreen')
            plt.plot(times, mean_actuals_by_time / mean_actuals_by_time)
            plt.plot(times, pred_cumulative_prop_by_time, color='black', alpha = 0.8)
            plt.legend([bp_preds["boxes"][0], bp_incurreds["boxes"][0]], ['Predictions', 'Incurreds'])
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
            
            plt.title('Aggregate OCLs (as proportion of actual) (manually adjusted negatives to 0)')
            plt.ylabel('Ratio')
            plt.show()

            
            # plotting weighted vsInc by claim size (q2 and q1-q3 bands)
            plt.plot(times, q2_weighted_vsInc_claimsize_by_time)
            plt.fill_between(times, q1_weighted_vsInc_claimsize_by_time, 
                            q3_weighted_vsInc_claimsize_by_time, alpha=0.3, color='blue')

            plt.title('Weighted vsInc (Claim Size) (q2 with q1-q3 bands)')
            plt.ylabel('Weighted vsInc (Claim Size) (%)')
            plt.grid(axis='both', linestyle='--', alpha=0.7)

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

            # boxplot of weighted vsInc by claim size
            box_plot(weighted_vsInc_claimsize_by_time, positions=times, median_colour='fuchsia', edge_colour='darkblue', fill_colour='lightblue')
            plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)
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

            # plotting weighted vsInc by ocl (q2 and q1-q3 bands)
            plt.plot(times, q2_weighted_vsInc_ocl_by_time)
            plt.fill_between(times, q1_weighted_vsInc_ocl_by_time, 
                            q3_weighted_vsInc_ocl_by_time, alpha=0.3, color='blue')
            
            plt.title('Weighted vsInc (OCL) (q2 with q1-q3 bands)')
            plt.ylabel('Weighted vsInc (OCL) (%)')
            plt.grid(axis='both', linestyle='--', alpha=0.7)

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
            box_plot(weighted_vsInc_ocl_by_time, positions=times, median_colour='fuchsia', edge_colour='darkblue', fill_colour='lightblue')
            plt.plot(times, pred_cumulative_prop_by_time * 100, color='black', alpha = 0.7)
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

        # plotting vsInc
        plt.plot(times, mean_vsInc_by_time, label='vsInc')
        plt.fill_between(times, mean_vsInc_by_time - sd_vsInc_by_time, 
                        mean_vsInc_by_time + sd_vsInc_by_time, alpha=0.3, color='blue')
        plt.fill_between(times, mean_vsInc_by_time - 2*sd_vsInc_by_time, 
                        mean_vsInc_by_time + 2*sd_vsInc_by_time, alpha=0.2, color='blue')
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
        plt.plot(times, mean_weighted_vsInc_claimsize_by_time, label='Weighted vsInc (Claim Size)')
        plt.fill_between(times, mean_weighted_vsInc_claimsize_by_time - sd_weighted_vsInc_claimsize_by_time, 
                        mean_weighted_vsInc_claimsize_by_time + sd_weighted_vsInc_claimsize_by_time, alpha=0.3, color='blue')
        plt.fill_between(times, mean_weighted_vsInc_claimsize_by_time - 2*sd_weighted_vsInc_claimsize_by_time, 
                        mean_weighted_vsInc_claimsize_by_time + 2*sd_weighted_vsInc_claimsize_by_time, alpha=0.2, color='blue')
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
        plt.plot(times, mean_weighted_vsInc_ocl_by_time, label='Weighted vsInc (OCL)')
        plt.fill_between(times, mean_weighted_vsInc_ocl_by_time - sd_weighted_vsInc_ocl_by_time,
                        mean_weighted_vsInc_ocl_by_time + sd_weighted_vsInc_ocl_by_time, alpha=0.3, color='blue')
        plt.fill_between(times, mean_weighted_vsInc_ocl_by_time - 2*sd_weighted_vsInc_ocl_by_time,
                        mean_weighted_vsInc_ocl_by_time + 2*sd_weighted_vsInc_ocl_by_time, alpha=0.2, color='blue')
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

def cross_validate(fp_in, fp_out, hyperparameter_grid, verbose=True):
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
                                  hp_comb['model_type'])
        
        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_in + 'val_index.csv', 
                                fp_in + 'val_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'])
        
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

def train_multiple_initialisations(fp_in, fp_out, hp_comb, iterations, verbose=True):
    '''Retrains the model on the test set multiple times, producing graphical 
       summaries for the first iteration as well as some graphical summaries 
       of the distribution of predictions
       
       Args:
         fp_in: filepath to the folder with the test indexes and sets
         fp_out: filepath to the folder that stores the trained model weights
         hp_comb: dictionary of hyperparameters
         iterations: number of times to retrain the model
         verbose: whether to print written outputs and progress to console'''

    train_set = ClaimsDataset(hp_comb['target_col'], 
                              fp_in + 'train_index.csv', 
                              fp_in + 'train_set.csv', 
                              hp_comb['include_incurreds'],
                              hp_comb['include_covariates'],
                              hp_comb['transform_inputs'],
                              hp_comb['model_type'])

    val_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'val_index.csv', 
                            fp_in + 'val_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'])

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

def test_multiple_initialisations(fp_in, fp_out, hp_comb, iterations, verbose=True):
    test_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'test_index.csv', 
                            fp_in + 'test_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'],
                            hp_comb['model_type'])
    
    preds_matrix = np.zeros(shape=(iterations, len(test_set.index.index)))
    vsInc_list = np.zeros(shape=iterations)
    weighted_vsInc_claimsize_list = np.zeros(shape=iterations)
    weighted_vsInc_ocl_list = np.zeros(shape=iterations)

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

        preds_matrix[i] = preds
        vsInc_list[i] = get_vsInc(actuals, preds, incurreds)
        weighted_vsInc_claimsize_list[i] = get_weighted_vsInc_claimsize(actuals, preds, 
                                                    incurreds)
        
        weighted_vsInc_ocl_list[i] = get_weighted_vsInc_ocl(actuals, preds, incurreds, ocls)
        

    # manually capping claim size predictions
    # largest in one of the datasets was $6m, so setting cap at $100m
    preds_matrix[preds_matrix > 1e8] = 1e8


    # Assessing distribution of aggregate claims
    aggregate_preds = preds_matrix.sum(axis=1)
    aggregate_actuals = actuals.sum()
    aggregate_incurreds = incurreds.sum()

    val_date = 40

    # finding aggregate incurred at the valuation date (calendar quarter 40)
    preds_val = preds_matrix[:, test_set.index['pred_time'] == val_date]
    actuals_val = actuals[test_set.index['pred_time'] == val_date]
    #print(f'actuals_val: {actuals_val}')
    incurreds_val = incurreds[test_set.index['pred_time'] == val_date]
    #print(f'incurreds_val: {incurreds_val}')
    ocls_val = ocls[test_set.index['pred_time'] == val_date]

    aggregate_preds_val = preds_val.sum(axis=1)
    aggregate_actuals_val = actuals_val.sum()
    aggregate_incurreds_val = incurreds_val.sum()

    # finding total paid and ocl at the valuation date
    test_paid = test_set.set.loc[test_set.set['pred_time'] == val_date, 
                                 ['claim_no', 'paid']].groupby(['claim_no'])['paid'].max()

    #print(f'test_paid: {test_paid}')

    val_date_paid = test_paid.sum()

    #print(f'actual ocls: {actuals_val - test_paid.values}')
    model_ocls_run1 = preds_val[0] - test_paid.values
    #print(f'model ocls (1st run): {model_ocls_run1}')
    #print(f'incurred ocls: {incurreds_val - test_paid.values}')

    print(f'proportion of negative model ocls at valuation date: {np.mean((model_ocls_run1) < 0)}')
    print(f'negative model ocls at valuation date: {(model_ocls_run1)[(model_ocls_run1) < 0]}')
    print(f'sum of negative model ocls at valuation date: {np.sum((model_ocls_run1)[(model_ocls_run1) < 0])}')
    print(f'sum of actual ocls from preds with negative ocl: {np.sum((actuals_val - test_paid.values)[(model_ocls_run1) < 0])}')
    print(f'sum of all actual ocls: {np.sum(actuals_val - test_paid.values)}\n')

    print(f'proportion of negative model claim sizes at valuation date: {np.mean(preds_val[0] < 0)}')
    print(f'negative model claim sizes at valuation date: {preds_val[0][preds_val[0] < 0]}')
    print(f'sum of negative model claim sizes at valuation date: {np.sum(preds_val[0][preds_val[0] < 0])}\n')


    paids = test_set.set.loc[:,['claim_no', 'pred_time', 'paid']].groupby(['claim_no', 'pred_time'])['paid'].max()
    model_ocls = preds_matrix[0] - paids

    print(f"proportion of negative model OCLs: {np.mean(model_ocls < 0)}")
    print(f'sum of negative model OCLs: {np.sum((model_ocls)[(model_ocls) < 0])}')
    actuals_copy = actuals.copy().reset_index(drop=True)
    paids_copy = paids.copy().reset_index(drop=True)
    model_ocls_copy = model_ocls.copy().reset_index(drop=True)
    print(f'sum of actual OCLs from preds with negative OCL: {np.sum((actuals_copy - paids_copy)[(model_ocls_copy) < 0])}')
    print(f'sum of all actual OCLs: {np.sum(actuals_copy - paids_copy)}')


    ocl_preds = aggregate_preds_val - [val_date_paid] * len(aggregate_preds_val)
    ocl_actuals = aggregate_actuals_val - val_date_paid
    ocl_incurreds = aggregate_incurreds_val - val_date_paid

    # weighted vsInc at the valuation date
    weighted_vsInc_claimsize_list_val = np.array([get_weighted_vsInc_claimsize(actuals_val, 
                                                           preds, 
                                                           incurreds_val)
                                        for preds in preds_val])
    
    weighted_vsInc_ocl_list_val = np.array([get_weighted_vsInc_ocl(actuals_val,
                                                                   preds,
                                                                   incurreds_val,
                                                                   ocls_val)
                                        for preds in preds_val])

    # PLOTTING RESULTS ACROSS ALL WEIGHT INITIALISATIONS

    #print(f'type of actuals: {type(actuals)}')
    #print(f'type of preds_matrix: {type(preds_matrix)}')


    # Plotting aggregate claim size by dev quarter and cal quarter (mean across all iterations)
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'dev_quarter')
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'pred_time')
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'rept_quarter')
    aggregate_by_time(test_set.index, actuals, preds_matrix, incurreds, ocls, 'acc_quarter')

    # Histogram of aggregate claims across all prediction times
    plt.hist(aggregate_preds, weights=(np.zeros_like(aggregate_preds) + 1. / 
                                       aggregate_preds.size), color='thistle')
    

    plt.xlabel('Aggregate predictions')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of vsInc accuracy
    plt.hist(vsInc_list, weights=(np.zeros_like(vsInc_list) + 1. / 
                                  vsInc_list.size), color='lightgreen')
    
    plt.xlabel('vsInc accuracy (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of weighted vsInc (Claim Size)
    plt.hist(weighted_vsInc_claimsize_list, weights=(np.zeros_like(weighted_vsInc_claimsize_list) + 
                                           1. / weighted_vsInc_claimsize_list.size), 
                                           color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of distribution of weighted vsInc (OCL)
    plt.hist(weighted_vsInc_ocl_list, weights=(np.zeros_like(weighted_vsInc_ocl_list) +
                                             1. / weighted_vsInc_ocl_list.size),
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
    plt.hist(weighted_vsInc_claimsize_list_val, 
             weights=(np.zeros_like(weighted_vsInc_claimsize_list_val) + 
                      1. / weighted_vsInc_claimsize_list_val.size), 
             color='lightgreen')
    
    plt.xlabel('weighted vsInc (Claim Size) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of weighted vsInc (OCL) at the valuation date
    plt.hist(weighted_vsInc_ocl_list_val,
                weights=(np.zeros_like(weighted_vsInc_ocl_list_val) +
                            1. / weighted_vsInc_ocl_list_val.size),
                color='lightgreen')
    
    plt.xlabel('weighted vsInc (OCL) (%) at valuation date')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of OCL at the valuation date
    plt.hist(ocl_preds, weights=(np.zeros_like(ocl_preds) + 1. / 
                                       ocl_preds.size), color='thistle')
    
    plt.axvline(ocl_actuals, color='dodgerblue', linestyle='dashed', 
                linewidth=2)
    
    plt.axvline(ocl_incurreds, color='red', linestyle='dashed', 
                linewidth=2)

    plt.xlabel('OCL at valuation date')
    plt.ylabel('Frequency')
    plt.show()



def train_multiple_datasets(fp_py, fp_out, seed_base, max_iter, hp_comb):
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
                                hp_comb['model_type'])

        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'val_index.csv', 
                                fp_py_full + 'val_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'])

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

def test_multiple_datasets(fp_py, fp_out, seed_base, max_iter, hp_comb):
    results = []

    for i in range(1, max_iter + 1):
        fp_py_full = fp_py + str(i + seed_base) + '/'

        test_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_py_full + 'test_index.csv', 
                                fp_py_full + 'test_set.csv', 
                                hp_comb['include_incurreds'],
                                hp_comb['include_covariates'],
                                hp_comb['transform_inputs'],
                                hp_comb['model_type'])

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

        aggregate_preds_val = preds_val.sum()
        aggregate_actuals_val = actuals_val.sum()
        aggregate_incurreds_val = incurreds_val.sum()

        # finding total paid and ocl at the valuation date
        test_paid = test_set.set.loc[test_set.set['pred_time'] == val_date, 
                                    ['claim_no', 'paid']].groupby(['claim_no']).max()
        
        val_date_paid = test_paid.sum()['paid']

        ocl_preds_val = aggregate_preds_val - val_date_paid
        ocl_actuals_val = aggregate_actuals_val - val_date_paid
        ocl_incurreds_val = aggregate_incurreds_val - val_date_paid

        ocl_error_preds = round_threshold(abs(ocl_preds_val - ocl_actuals_val) / ocl_actuals_val)
        ocl_error_incurreds = round_threshold(abs(ocl_incurreds_val - ocl_actuals_val) / ocl_actuals_val)

        # weighted vsInc at the valuation date
        weighted_vsInc_claimsize_val = round_threshold(get_weighted_vsInc_claimsize(actuals_val, preds_val, incurreds_val))
        weighted_vsInc_ocl_val = round_threshold(get_weighted_vsInc_ocl(actuals_val, preds_val, incurreds_val, ocls_val))

        # MALE and MSLE
        MALE_preds = MeanAbsoluteLogError()(preds, actuals)
        MSLE_preds = MeanSquaredLogError()(preds, actuals)

        MALE_preds_val = MeanAbsoluteLogError()(preds_val, actuals_val)
        MSLE_preds_val = MeanSquaredLogError()(preds_val, actuals_val)
        
        MALE_incurreds = MeanAbsoluteLogError()(incurreds, actuals)
        MSLE_incurreds = MeanSquaredLogError()(incurreds, actuals)

        MALE_incurreds_val = MeanAbsoluteLogError()(incurreds_val, actuals_val)
        MSLE_incurreds_val = MeanSquaredLogError()(incurreds_val, actuals_val)

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
                        'ocl_actuals_val': ocl_actuals_val, 
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
    #print(results)

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
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'dev_quarter')
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'pred_time')
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'rept_quarter')
    aggregate_by_time(results['test_set_index'], actuals_matrix, preds_matrix, incurreds_matrix, ocls_matrix, 'acc_quarter')

    print('\nValuation Date Predictions:\n')
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'dev_quarter')
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'pred_time')
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'rept_quarter')
    aggregate_by_time(results['test_set_index_val'], actuals_matrix_val, preds_matrix_val, incurreds_matrix_val, ocls_matrix_val, 'acc_quarter')


    # Histogram of distribution of vsInc accuracy
    plt.hist(results['vsInc'], weights=(np.zeros_like(results['vsInc']) + 1. / 
                                  results['vsInc'].size), color='lightgreen')
    
    plt.xlabel('vsInc accuracy (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot of vsInc accuracy
    sns.boxplot(data=results['vsInc'], color='lightgreen')
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
    sns.boxplot(data=results['weighted_vsInc_claimsize'], color='skyblue')
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
    sns.boxplot(data=results['weighted_vsInc_ocl'], color='yellow')
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
    sns.boxplot(data=results['weighted_vsInc_claimsize_val'], color='skyblue')
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
    sns.boxplot(data=results['weighted_vsInc_ocl_val'], color='yellow')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('weighted vsInc (OCL) (%) at valuation date')
    plt.title('weighted vsInc (OCL) at valuation date')
    plt.show()

    # Boxplots of OCL errors at valuation date
    sns.boxplot(data=results[['ocl_error_preds_val', 'ocl_error_incurreds_val']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], ['Predictions', 'Incurreds'])
    plt.ylabel('OCL error (%)')
    plt.title('OCL errors at valuation date')
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
    sns.boxplot(data=results[['MALE_preds', 'MALE_incurreds']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], ['Predictions', 'Incurreds'])
    plt.title('MALE across multiple datasets')
    plt.ylabel('MALE')
    plt.show()

    sns.boxplot(data=results[['MSLE_preds', 'MSLE_incurreds']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], ['Predictions', 'Incurreds'])
    plt.title('MSLE across multiple datasets')
    plt.ylabel('MSLE')
    plt.show()

    # Boxplots of MALE and MSLE at valuation date
    sns.boxplot(data=results[['MALE_preds_val', 'MALE_incurreds_val']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], ['Predictions', 'Incurreds'])
    plt.title('MALE at valuation date')
    plt.ylabel('MALE')
    plt.show()

    sns.boxplot(data=results[['MSLE_preds_val', 'MSLE_incurreds_val']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([0, 1], ['Predictions', 'Incurreds'])
    plt.title('MSLE at valuation date')
    plt.ylabel('MSLE')
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

