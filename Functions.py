### IMPORTS ###################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from itertools import product

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

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
        return np.mean(np.abs(np.log(preds) - np.log(actuals)))
    
class MeanSquaredLogError(nn.Module):
    def __init__(self):
        super(MeanSquaredLogError, self).__init__()

    def forward(self, preds, actuals):
        return np.mean(np.square(np.log(preds) - np.log(actuals)))

### MODEL CLASSES #############################################################

class ClaimsDataset(Dataset):
    """ Based on Arkie's ClaimsDataset
    
    Notes: 
    - dataloader has to iterate from 0:len(dataset)
    - all sequences are padded to a minimum length of 40
    """

    def __init__(self, target_col, index_path, set_path, version_no=1, 
                 include_incurreds=True):
        self.target_col = target_col # string referring to name of 
        # target column (i.e. 'claim_size', 'log_m')
        self.index = pd.read_csv(index_path) 
        self.set = pd.read_csv(set_path)
        self.version_no = version_no # either 1, 2 or 3
        self.include_incurreds = include_incurreds # whether to use case 
        # estimate data or not

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

        nrows = df[(df['dev_time']!=0)\
                | (df['cal_time']!=0)\
                | (df['paid']!=0)\
                | (df['ocl']!=0)].shape[0]

        if self.version_no == 1:
            
            if self.include_incurreds:
                databox = torch.tensor(df[['dev_time', 'cal_time', 
                                           'paid', 'ocl']].values)
                
            else:
               databox = torch.tensor(df[['dev_time','cal_time','paid']].values) 

            # Return padded data
            return (torch.nn.functional.pad(databox.float(), (0,0,0,40-nrows)), 
                    target, claim_size, latest_incurred, true_ocl, real_index, 
                    claim_no, pred_time, nrows)

        elif self.version_no == 2:
            if self.include_incurreds:
                databox = torch.tensor(df[['dev_time', 'cal_time', 'paid', 'ocl', 
                                       'is_payment', 'is_major', 'is_minor', 
                                       'multiplier']].values)

            else:
                databox = torch.tensor(df[['dev_time', 'cal_time', 'paid', 
                                           'is_payment', 'is_major', 'is_minor', 
                                           'multiplier']].values)

            # Return padded data
            return (torch.nn.functional.pad(databox.float(), (0,0,0,40-nrows)), 
                    target, claim_size, latest_incurred, true_ocl, real_index, 
                    claim_no, pred_time, nrows)

        elif self.version_no == 3:
            num_payments = self.index['num_payments'][index]
            mean_payments = self.index['mean_payments'][index]
            var_payments = self.index['var_payments'][index]
            max_payment = self.index['max_payment'][index]
            num_revisions = self.index['num_revisions'][index]
            num_upward = self.index['num_upward'][index]
            total_variation = self.index['total_variation'][index]

            if self.include_incurreds:
                databox = torch.tensor(df[['dev_time', 'cal_time', 'paid', 'ocl', 
                                       'is_payment', 'is_major', 'is_minor', 
                                       'multiplier']].values)
                
            else:
                databox = torch.tensor(df[['dev_time', 'cal_time', 'paid', 
                                           'is_payment', 'is_major', 'is_minor', 
                                           'multiplier']].values)

            # Return padded data
            return (torch.nn.functional.pad(databox.float(), (0,0,0,40-nrows)),  
                    target, claim_size, latest_incurred, true_ocl, real_index, 
                    claim_no, pred_time, nrows, num_payments, mean_payments, 
                    var_payments, max_payment, num_revisions, num_upward, 
                    total_variation)

        else:
            raise ValueError('version_no must be 1, 2 or 3')
        
class ClaimsRNN(nn.Module):
    """
    Can use vanilla RNN, LSTM and GRU
    Can change this to experiement with different architectures/hyperparameters
    """

    def __init__(self, nHidden, nLayers, nOut, version_no=1, type='RNN', 
                 nonlinearity='relu', output_layer='linear', dropout=0.0, 
                 normalisation=None, include_incurreds=True):
        
        super(ClaimsRNN, self).__init__()
        self.nHidden = nHidden # number of hidden units
        self.nLayers = nLayers # number of layers
        self.version_no = version_no # either 1, 2 or 3
        self.type = type # either 'RNN', 'LSTM', or 'GRU'
        self.nonlinearity = nonlinearity # either 'relu' or 'tanh'
        # nonLinearity only used in vanilla RNN
        self.output_layer = output_layer # either 'linear' or 'exponential'
        self.dropout = dropout # float between 0 and 1
        self.include_incurreds = include_incurreds # needs to match ClaimsDataset
        self.relu = nn.ReLU() # used for feed-forward hidden layer, should change this so different activation functions can be specified
        # include_incurreds variable in ClaimsDataset

        # nFeatures is the number of features to be input into the RNN layer
        if self.version_no == 1:
            self.nFeatures = 3 + self.include_incurreds # 4 features with ocl, 3 without
        
        elif self.version_no == 2 or self.version_no == 3:
            self.nFeatures = 7 + self.include_incurreds # 8 features with ocl, 7 without

        # currently batch normalisation has been implemented, can add others
        if normalisation == 'batch':
            self.normalisation = nn.BatchNorm1d(self.nFeatures)

        else:
            self.normalisation = None

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

        if version_no == 3:
            # +8 becuase we are adding 7 derived features into the fc layer + 1 for pred time
            self.fc1 = nn.Linear(nHidden + 8, nHidden // 2) 

        else:
            # +1 for pred time
            self.fc1 = nn.Linear(nHidden + 1, nHidden // 2)

        self.fc2 = nn.Linear(nHidden // 2, nOut)


    def forward(self, x):

        if self.normalisation == 'batch':
            x = self.normalisation(x)

        if self.version_no == 3:
            # x will be a tuple with the first element being the datapoints 
            # and the rest being the extra features going into the fc layer
            out, ht = self.rnn(x[0])

            if self.type == 'LSTM':
                ht = ht[0]

            # Concatenating the extra features
            out = torch.cat((ht[-1,:,:], 
                             x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]), 1)

            # Layer normalisation to ensure derived features have
            # similar scale to RNN outputs
            layer_norm = nn.LayerNorm(out.size()).to(device)
            out = layer_norm(out)

        else:
            # x will be the packed datapoints
            out, ht = self.rnn(x[0])

            if self.type == 'LSTM':
                ht = ht[0]

            out = torch.cat((ht[-1,:,:], x[1]), 1)

        out = self.fc1(out)

        # non-linear activation for feed-forward hidden layer
        out = self.relu(out)

        out = self.fc2(out)

        if self.output_layer == 'exponential':
            out = torch.exp(out)

        return out

### TRAINING/TESTING FUNCTIONS ################################################

def train_network(model, train_data, hp_comb, optimiser, verbose=True, 
                  val_data=None, cv_loss_list=None, cv_vsInc_list=None, cv_uie_list=None):
    
    """
    Args:
        model: the model to train
        dataset: ClaimsDataset object containing training data
        epochs: the maximum number of epochs to train for
        batch_size: the batch size
        optimiser: the optimiser to use
        criterion: loss function
        version_no: input set (1, 2 or 3)
        verbose: whether to print written outputs and progress to console
        val_data: ClaimsDataset object containing validation data, 
                  enables early stopping
        patience: max number of epochs without improvements in validation loss
        cv_loss_list/cv_vsInc_list/cv_uie_list: lists to be passed to keep 
        track of between-model stats during cross-validation
    
    """

    if val_data is not None:
        val_loss_list = []
        val_vsInc_list = []
        val_uie_list = []
        best_val_loss = np.inf
        best_val_vsInc = 0
        best_val_uie = np.inf
        best_weights = None
        patience_counter = 0

    # Data loader
    trainloader = torch.utils.data.DataLoader(dataset=train_data, 
                                              batch_size=hp_comb['batch_size'], 
                                              shuffle=True, drop_last=True)

    # Train the model
    for epoch in range(hp_comb['epochs']):
        total_loss = 0
        total_datapoints = 0
        total_vsInc = 0
        total_uie = 0
        total_weighted_vsinc = 0
        total_observation_sizes = 0

        for batch in trainloader:

            # extract batch data
            if hp_comb['version_no'] == 3:
                (datapoints, targets, claim_sizes, latest_incurreds, true_ocls,
                 indexes, claim_nos, pred_times, nrowss, num_paymentss, 
                 mean_paymentss, var_paymentss, max_payments, num_revisionss, 
                 num_upwards, total_variations) = batch

                num_paymentss = num_paymentss.unsqueeze(1).to(device).float()
                mean_paymentss = mean_paymentss.unsqueeze(1).to(device).float()
                var_paymentss = var_paymentss.unsqueeze(1).to(device).float()
                max_payments = max_payments.unsqueeze(1).to(device).float()
                num_revisionss = num_revisionss.unsqueeze(1).to(device).float()
                num_upwards = num_upwards.unsqueeze(1).to(device).float()
                total_variations=total_variations.unsqueeze(1).to(device).float()

            else:
                (datapoints, targets, claim_sizes, latest_incurreds, true_ocls,
                 indexes, claim_nos, pred_times, nrowss) = batch
                
            datapoints = datapoints.to(device).float()
            targets = targets.to(device).float()
            claim_sizes = claim_sizes.to(device).float()
            latest_incurreds = latest_incurreds.to(device).float()
            true_ocls = true_ocls.to(device).float()
            pred_times = pred_times.unsqueeze(1).to(device).float()

            packed = pack_padded_sequence(datapoints, nrowss, 
                                          enforce_sorted=False, 
                                          batch_first=True)

            if hp_comb['version_no'] == 3:
                # create a tuple with packed and the extra features
                packed_extra = (packed, pred_times, num_paymentss, 
                                mean_paymentss, var_paymentss, max_payments, 
                                num_revisionss, num_upwards, total_variations)

            else:
                packed_extra = (packed, pred_times)  
                
            raw_preds = model(packed_extra)    
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            # converting raw preds and targets to be in terms of ultimate claim size
            if train_data.target_col == 'claim_size':
                preds = raw_preds
                ultimates = targets
            
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
                ValueError('Invalid target, must be "claim_size", "log_m", "true_ocl" or "log_true_ocl"')


            # Loss and gradient descent
            loss = hp_comb['criterion'](raw_preds, targets)
            optimiser.zero_grad()
            loss.backward() # Calculate gradients
            optimiser.step() # Update weights

            # Track statistics
            total_loss += loss.item() * preds.size(0)
            total_datapoints += preds.size(0)
            total_vsInc += sum(torch.abs((ultimates-preds)) < 
                               torch.abs((ultimates-latest_incurreds)))
            
            total_weighted_vsinc += sum(ultimates * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_observation_sizes += sum(ultimates)
            
            total_uie+=sum(torch.logical_and((preds < latest_incurreds), 
                                             (torch.abs((ultimates-preds)) > 
                                              torch.abs((ultimates-
                                                         latest_incurreds)))))

        # End of epoch summary
        vs_incurred_accuracy = total_vsInc / total_datapoints * 100
        uie = total_uie / total_datapoints * 100
        total_loss = total_loss / total_datapoints
        weighted_vsinc = total_weighted_vsinc / total_observation_sizes * 100

        if verbose:
            print(f'Epoch {epoch}: '
                  f'training loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc = {weighted_vsinc:.2f}%, '
                  f'UIE = {uie:.2f}%')

        # Validation
        if val_data:
            
            if verbose:
                print('Validation')
                
            test_network(model, val_data, hp_comb, val_loss_list=val_loss_list, 
                         val_vsInc_list=val_vsInc_list, 
                         val_uie_list=val_uie_list, verbose=verbose)

            # Early stopping
            if val_loss_list[-1] < best_val_loss:
                best_val_loss = val_loss_list[-1]
                best_val_vsInc = val_vsInc_list[-1].item()
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
                if cv_uie_list is not None:
                    cv_uie_list.append(best_val_uie)
                
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch}')
                    print(f'Validation: '
                          f'loss = {round_threshold(best_val_loss):,}, '
                          f'vsInc = {best_val_vsInc:.2f}%, '
                          f'UIE = {best_val_uie:.2f}%\n')
                    
                break

    # if we reach max number of epochs, save the best weights
    if ((epoch == hp_comb['epochs'] - 1) and 
        (val_data is not None) and 
        (cv_loss_list is not None) and 
        (cv_vsInc_list is not None)):

        model.load_state_dict(best_weights)
        cv_loss_list.append(best_val_loss)
        cv_vsInc_list.append(best_val_vsInc)
        cv_uie_list.append(best_val_uie)

        if verbose:
            print(f'\nNo early stopping')
            print(f'Validation: loss = {round_threshold(best_val_loss):,}, '
                  f'vsInc = {best_val_vsInc:.2f}%, '
                  f'UIE = {best_val_uie:.2f}%\n')
            
def test_network(model, test_data, hp_comb, preds_list=None, verbose=True, 
                 val_loss_list=None, val_vsInc_list=None, val_uie_list=None):
    
    """Args:
        preds_list: empty list to append predictions to
        val_loss_list/val_vsInc_list/val_uie_list: lists to be passed to keep 
        track of within-model stats during training
    """

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                              batch_size=hp_comb['batch_size'], 
                                              shuffle=False)

    total_loss = 0
    total_datapoints = 0
    total_vsInc = 0
    total_uie = 0
    total_weighted_vsinc = 0
    total_observation_sizes = 0

    # Test the model
    with torch.no_grad():
        for batch in test_loader:
            if hp_comb['version_no'] == 3:
                (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                 indexes, claim_nos, pred_times, nrowss, num_paymentss, 
                 mean_paymentss, var_paymentss, max_payments, num_revisionss, 
                 num_upwards, total_variations) = batch

                num_paymentss = num_paymentss.unsqueeze(1).to(device).float()
                mean_paymentss = mean_paymentss.unsqueeze(1).to(device).float()
                var_paymentss = var_paymentss.unsqueeze(1).to(device).float()
                max_payments = max_payments.unsqueeze(1).to(device).float()
                num_revisionss = num_revisionss.unsqueeze(1).to(device).float()
                num_upwards = num_upwards.unsqueeze(1).to(device).float()
                total_variations = total_variations.unsqueeze(1).to(device).float()

            else:
                (datapoints, targets, claim_sizes, latest_incurreds, true_ocls, 
                 indexes, claim_nos, pred_times, nrowss) = batch

            datapoints = datapoints.to(device).float()
            targets = targets.to(device).float()
            claim_sizes = claim_sizes.to(device).float()
            latest_incurreds = latest_incurreds.to(device).float()
            true_ocls = true_ocls.to(device).float()
            pred_times = pred_times.unsqueeze(1).to(device).float()

            packed = pack_padded_sequence(datapoints, nrowss, 
                                          enforce_sorted=False, 
                                          batch_first=True)

            if hp_comb['version_no'] == 3:
                # create a tuple with packed and the extra features
                packed_extra = (packed, pred_times, num_paymentss, 
                                mean_paymentss, var_paymentss, max_payments, 
                                num_revisionss, num_upwards, total_variations)

            else:
                packed_extra = (packed, pred_times)

            raw_preds = model(packed_extra)
            raw_preds = raw_preds.reshape(raw_preds.shape[0])

            if test_data.target_col == 'claim_size':
                preds = raw_preds
                ultimates = targets
            
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
                ValueError('Invalid target, must be "claim_size", "log_m", "true_ocl" or "log_true_ocl"')


            # Loss and gradient descent
            loss = hp_comb['criterion'](raw_preds, targets)

            # Track statistics
            total_loss += loss.item() * preds.size(0)
            total_datapoints += preds.size(0)
            total_vsInc += sum(torch.abs((ultimates-preds)) < 
                               torch.abs((ultimates-latest_incurreds)))
            
            total_weighted_vsinc += sum(ultimates * (torch.abs((ultimates-preds)) < 
                                        torch.abs((ultimates-latest_incurreds))))
            
            total_observation_sizes += sum(ultimates)
            
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
        weighted_vsinc = total_weighted_vsinc / total_observation_sizes * 100

        if verbose:
            print(f'loss = {round_threshold(total_loss):,}, '
                  f'vsInc = {vs_incurred_accuracy:.2f}%, '
                  f'weighted vsInc = {weighted_vsinc:.2f}%, '
                  f'UIE = {uie:.2f}%')

        if isinstance(val_loss_list, list):
            val_loss_list.append(total_loss)

        if isinstance(val_vsInc_list, list):
            val_vsInc_list.append(vs_incurred_accuracy)

        if isinstance(val_uie_list, list):
            val_uie_list.append(uie)

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

def get_losses(actuals, preds, incurreds):
    '''Computes and prints the MALE and MSLE for the model's predictions and 
    the case estimates.
    
    The losses will be calculated using ultimate claim size for 'claim_size' 
    and 'log_m' targets, and true ocl for 'true_ocl' and 'log_true_ocl' targets.'''

    preds_male = MeanAbsoluteLogError()(preds, actuals)
    preds_msle = MeanSquaredLogError()(preds, actuals)

    incurreds_male = MeanAbsoluteLogError()(incurreds, actuals)
    incurreds_msle = MeanSquaredLogError()(incurreds, actuals)

    print(f'model MALE: {preds_male:.3f}, MSLE: {preds_msle:.3f}')
    print(f'incurred MALE: {incurreds_male:.3f}, MSLE: {incurreds_msle:.3f}')

def get_vsInc(actuals, preds, incurreds):
    return 100 * np.mean(np.abs((actuals-preds)) < np.abs((actuals-incurreds)))

def get_weighted_vsInc(actuals, preds, incurreds):
    return 100 * np.average(np.abs((actuals-preds)) < 
                            np.abs((actuals-incurreds)), weights=actuals)

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
                 val_loss_list=None, val_vsInc_list=None, val_uie_list=None)

    preds_list = pd.Series(preds_list)
    actuals_list = test_data.index["claim_size"]
    incurreds_list = test_data.index["latest_incurred"]

    return actuals_list, preds_list, incurreds_list

def get_latest(test_data, actuals, preds, incurreds):
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

    return latest_actuals, latest_preds, latest_incurreds

def get_dev_quarter(test_data, actuals, preds, incurreds, dev_quarter):
    '''Finds the predictions for a specific development quarter and returns the
       model predictions, and associated actuals and case estimates'''
    indicator = test_data.index['dev_quarter'] == dev_quarter
    dev_actuals = actuals[indicator]
    dev_preds = preds[indicator]
    dev_incurreds = incurreds[indicator]

    return dev_actuals, dev_preds, dev_incurreds

def aggregate_by_time(index_data, actuals, preds, incurreds, time_str):
    '''Plots aggregate claims, vsInc and weighted vsInc over time, 
       either calendar or development'''

    times = np.sort(index_data[time_str].unique())
    
    actuals_by_time = np.zeros(len(times))
    preds_by_time = np.zeros(len(times))
    incurreds_by_time = np.zeros(len(times))
    vsInc_by_time = np.zeros(len(times))
    weighted_vsInc_by_time = np.zeros(len(times))
    
    for index, time in enumerate(times):
        indicator = index_data[time_str] == time
        actuals_by_time[index] = np.sum(actuals[indicator])
        preds_by_time[index] = np.sum(preds[indicator])
        incurreds_by_time[index] = np.sum(incurreds[indicator])
        vsInc_by_time[index] = get_vsInc(actuals[indicator], 
                                         preds[indicator], 
                                         incurreds[indicator])
        
        weighted_vsInc_by_time[index] = get_weighted_vsInc(actuals[indicator], 
                                                           preds[indicator], 
                                                           incurreds[indicator])

    # plotting aggregate preds
    plt.plot(times, actuals_by_time, label='Actuals')
    plt.plot(times, preds_by_time, label='Predictions')
    plt.plot(times, incurreds_by_time, label='Incurreds')
    plt.legend(loc='upper right')
    plt.ylabel('Aggregate claims')

    if time_str == 'pred_time':
        plt.xlabel('Calendar quarter')
    elif time_str == 'dev_quarter':
        plt.xlabel('Development quarter')
        
    plt.show()

    # plotting vsInc
    plt.plot(times, vsInc_by_time)
    plt.ylabel('vsInc (%)')

    if time_str == 'pred_time':
        plt.xlabel('Calendar quarter')
    elif time_str == 'dev_quarter':
        plt.xlabel('Development quarter')
        
    plt.show()

    # plotting weighted vsInc
    plt.plot(times, weighted_vsInc_by_time)
    plt.ylabel('Weighted vsInc (%)')

    if time_str == 'pred_time':
        plt.xlabel('Calendar quarter')
    elif time_str == 'dev_quarter':
        plt.xlabel('Development quarter')
        
    plt.show()
    
def get_aggregates(actuals, preds, incurreds):
    '''Prints the sum over each claim and censor point for all claims, 
       predictions and case estimates'''

    aggregate_preds = np.sum(preds)
    aggregate_actual = np.sum(actuals)
    aggregate_incurred = np.sum(incurreds)

    print(f'Aggregate predictions: {aggregate_preds:,.0f}')
    print(f'Aggregate actual: {aggregate_actual:,.0f}')
    print(f'Aggregate incurred: {aggregate_incurred:,.0f}')

def get_small_large(actuals, preds, incurreds, small_threshold, 
                    large_threshold):
    
    '''Splits the data into small, medium and large claims based on the 
    thresholds'''
    
    small_actuals = actuals[actuals < small_threshold]
    small_preds = preds[actuals < small_threshold]
    small_incurreds = incurreds[actuals < small_threshold]

    medium_actuals = actuals[(actuals > small_threshold) & 
                             (actuals < large_threshold)]
    
    medium_preds = preds[(actuals > small_threshold) & 
                         (actuals < large_threshold)]
    
    medium_incurreds = incurreds[(actuals > small_threshold) & 
                                 (actuals < large_threshold)]

    large_actuals = actuals[actuals >= large_threshold]
    large_preds = preds[actuals >= large_threshold]
    large_incurreds = incurreds[actuals >= large_threshold]

    return (small_actuals, small_preds, small_incurreds, 
            medium_actuals, medium_preds, medium_incurreds, 
            large_actuals, large_preds, large_incurreds)

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
    actuals, preds, incurreds = get_preds_actuals(model, dataset, hp_comb)
    get_aggregates(actuals, preds, incurreds)
    get_losses(actuals, preds, incurreds)
    print(f'vsInc: {get_vsInc(actuals, preds, incurreds):.2f}%')
    print(f'Weighted vsInc: {get_weighted_vsInc(actuals, preds, incurreds):.2f}%')
    print(f'number of preds: {len(preds)}')
    get_heatmap(actuals, preds, nbins=50)
    get_close_far(actuals, preds, incurreds)


    # Analysing the latest prediction for each claim
    print('Latest')
    latest_actuals, latest_preds, latest_incurreds = get_latest(dataset, 
                                                                actuals, 
                                                                preds, 
                                                                incurreds)
    
    get_aggregates(latest_actuals, latest_preds, latest_incurreds)
    get_losses(latest_actuals, latest_preds, latest_incurreds)
    print(f'vsInc: {get_vsInc(latest_actuals, latest_preds, latest_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(latest_actuals, 
                                        latest_preds, 
                                        latest_incurreds)
    
    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')

    print(f'number of preds: {len(latest_preds)}')
    get_heatmap(latest_actuals, latest_preds, nbins=40)
    get_close_far(latest_actuals, latest_preds, latest_incurreds)

    aggregate_by_time(dataset.index, actuals, preds, incurreds, 'pred_time')
    aggregate_by_time(dataset.index, actuals, preds, incurreds, 'dev_quarter')


    # Analysing all claims by the specified development quarters
    dev_quarters = [1, 5, 10, 16] # can adjust these to analyse different periods
    nbinss = [40, 30, 30, 20]
    for i in range(len(dev_quarters)):
        print(f'Dev Quarter {dev_quarters[i]}')
        (dev_actuals, 
         dev_preds, 
         dev_incurreds) = get_dev_quarter(dataset, actuals, preds, incurreds, 
                                          dev_quarter=dev_quarters[i])
        
        get_aggregates(dev_actuals, dev_preds, dev_incurreds)
        get_losses(dev_actuals, dev_preds, dev_incurreds)
        print(f'vsInc: {get_vsInc(dev_actuals, dev_preds, dev_incurreds):.2f}%')

        weighted_vsinc = get_weighted_vsInc(dev_actuals, 
                                            dev_preds, 
                                            dev_incurreds)
        
        print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
        
        print(f'number of preds: {len(dev_preds)}')
        get_heatmap(dev_actuals, dev_preds, nbins=nbinss[i])


    # Analysing all claims by size
    (small_actuals, small_preds, small_incurreds, 
     medium_actuals, medium_preds, medium_incurreds, 
     large_actuals, large_preds, 
     large_incurreds) = get_small_large(actuals, preds, incurreds, 
                                        small_threshold, large_threshold)
    
    print('Small')
    get_aggregates(small_actuals, small_preds, small_incurreds)
    get_losses(small_actuals, small_preds, small_incurreds)
    print(f'vsInc: {get_vsInc(small_actuals, small_preds, small_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(small_actuals, 
                                        small_preds, 
                                        small_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
    print(f'number of preds: {len(small_preds)}')
    get_heatmap(small_actuals, small_preds, nbins=30)
    get_close_far(small_actuals, small_preds, small_incurreds)

    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds, 
                                                           'pred_time')
    
    aggregate_by_time(dataset.index.loc[dataset.index['claim_size'] < 
                                        small_threshold,], small_actuals, 
                                                           small_preds, 
                                                           small_incurreds, 
                                                           'dev_quarter')

    print('Medium')
    get_aggregates(medium_actuals, medium_preds, medium_incurreds)
    get_losses(medium_actuals, medium_preds, medium_incurreds)
    print(f'vsInc: {get_vsInc(medium_actuals, medium_preds, medium_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(medium_actuals, 
                                        medium_preds, 
                                        medium_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
    print(f'number of preds: {len(medium_preds)}')
    get_heatmap(medium_actuals, medium_preds, nbins=40)
    get_close_far(medium_actuals, medium_preds, medium_incurreds)

    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, 'pred_time')
    
    aggregate_by_time(
        dataset.index.loc[(dataset.index['claim_size'] > small_threshold) & 
                          (dataset.index['claim_size'] < large_threshold),], 
        medium_actuals, medium_preds, medium_incurreds, 'dev_quarter')

    print('Large')
    get_aggregates(large_actuals, large_preds, large_incurreds)
    get_losses(large_actuals, large_preds, large_incurreds)
    print(f'vsInc: {get_vsInc(large_actuals, large_preds, large_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(large_actuals, 
                                        large_preds, 
                                        large_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
    print(f'number of preds: {len(large_preds)}')
    get_heatmap(large_actuals, large_preds,nbins=30)
    get_close_far(large_actuals, large_preds, large_incurreds)

    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, 'pred_time')
    
    aggregate_by_time(dataset.index[dataset.index['claim_size'] > large_threshold], 
                      large_actuals, large_preds, large_incurreds, 'dev_quarter')


    # Analysing latest predictions by ultimate size of claim
    (small_actuals, small_preds, small_incurreds, 
     medium_actuals, medium_preds, medium_incurreds, 
     large_actuals, large_preds, 
     large_incurreds) = get_small_large(latest_actuals, latest_preds, 
                                        latest_incurreds, small_threshold, 
                                        large_threshold)
    
    print('Small Latest')
    get_aggregates(small_actuals, small_preds, small_incurreds)
    get_losses(small_actuals, small_preds, small_incurreds)
    print(f'vsInc: {get_vsInc(small_actuals, small_preds, small_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(small_actuals, 
                                        small_preds, 
                                        small_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
    print(f'number of preds: {len(small_preds)}')
    get_heatmap(small_actuals, small_preds, nbins=10)
    get_close_far(small_actuals, small_preds, small_incurreds)

    print('Medium Latest')
    get_aggregates(medium_actuals, medium_preds, medium_incurreds)
    get_losses(medium_actuals, medium_preds, medium_incurreds)
    print(f'vsInc: {get_vsInc(medium_actuals, medium_preds, medium_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(medium_actuals, 
                                        medium_preds, 
                                        medium_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
    print(f'number of preds: {len(medium_preds)}')
    get_heatmap(medium_actuals, medium_preds, nbins=30)
    get_close_far(medium_actuals, medium_preds, medium_incurreds)

    print('Large Latest')
    get_aggregates(large_actuals, large_preds, large_incurreds)
    get_losses(large_actuals, large_preds, large_incurreds)
    print(f'vsInc: {get_vsInc(large_actuals, large_preds, large_incurreds):.2f}%')

    weighted_vsinc = get_weighted_vsInc(large_actuals, 
                                        large_preds, 
                                        large_incurreds)

    print(f'Weighted vsInc: {weighted_vsinc:.2f}%')
    
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

    NOTE: do not change loss function within 1 run of this function becuase 
          the loss numbers will be on a different scale.
    Same with switching from 'claim_size' to 'log_m'
    """

    best_val_loss = np.Inf
    best_val_vsInc = 0
    best_val_uie = np.Inf
    best_hp_comb = None
    
    for hp_comb in hyperparameter_grid:
        if verbose:
            print(f'\nTrying hyperparameter combination: {hp_comb}')

        cv_loss_list = []
        cv_vsInc_list = []
        cv_uie_list = []

        # Use the same seed for each hyperparameter combination so they 
        # all start from the same initial weights
        # Also means results can be reproduced for the best combination without
        # having to rerun the entire hyperparamter tuning process
        torch.manual_seed(1)

        train_set = ClaimsDataset(hp_comb['target_col'], 
                                  fp_in + 'train_index.csv', 
                                  fp_in + 'train_set.csv',
                                  hp_comb['version_no'],
                                  hp_comb['include_incurreds'])
        
        val_set = ClaimsDataset(hp_comb['target_col'], 
                                fp_in + 'val_index.csv', 
                                fp_in + 'val_set.csv', 
                                hp_comb['version_no'],
                                hp_comb['include_incurreds'])
        
        model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                          hp_comb['nOut'], hp_comb['version_no'], 
                          hp_comb['type'], hp_comb['nonlinearity'], 
                          hp_comb['output_layer'], hp_comb['dropout'], 
                          hp_comb['normalisation'], hp_comb['include_incurreds']).to(device)
        
        optimiser = optim.Adam(model.parameters(), lr=hp_comb['lr'])

        train_network(model, train_set, hp_comb, optimiser, verbose, val_set, 
                      cv_loss_list, cv_vsInc_list, cv_uie_list)
        
        cv_loss = np.mean(cv_loss_list)
        cv_vsInc = np.mean(cv_vsInc_list)
        cv_uie = np.mean(cv_uie_list)

        # 'best' model chosen based on val loss
        if cv_loss < best_val_loss:
            best_val_loss = cv_loss
            best_val_vsInc = cv_vsInc
            best_val_uie = cv_uie
            best_hp_comb = hp_comb
            best_weights = deepcopy(model.state_dict())
            print(f'\nnew best val_loss: {round_threshold(best_val_loss):,}, '
                  f'val_vsInc: {best_val_vsInc:.2f}%, '
                  f'val_uie = {best_val_uie:.2f}%\n')
        
        # appending results to dataframe
        row = pd.DataFrame({'dataset': fp_in.split('/')[-2], 
                            'include_incurreds': hp_comb['include_incurreds'],
                            'version_no': hp_comb['version_no'], 
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
                            'learning_rate': hp_comb['lr'], 
                            'normalisation': hp_comb['normalisation'], 
                            'dropout': hp_comb['dropout'], 
                            'loss': round_threshold(cv_loss), 
                            'vsInc': round(cv_vsInc, 2), 
                            'UIE': round(cv_uie, 2)}, index=[0])
        
        row.to_csv(fp_out, mode='a', header=False, index=False)

    if verbose:
        print(f'\nBest hyperparameter combination: {best_hp_comb}')
        print(f'Best validation loss: {round_threshold(best_val_loss):,}')
        print(f'Best validation vsInc: {best_val_vsInc:.2f}%')
        print(f'Best validation UIE: {best_val_uie:.2f}%')

    # Results for best model
    model = ClaimsRNN(best_hp_comb['nHidden'], best_hp_comb['nLayers'], 
                      best_hp_comb['nOut'], best_hp_comb['version_no'], 
                      best_hp_comb['type'], best_hp_comb['nonlinearity'], 
                      best_hp_comb['output_layer'], best_hp_comb['dropout'], 
                      best_hp_comb['normalisation'], best_hp_comb['include_incurreds']).to(device)
    
    model.load_state_dict(best_weights)
    analyse_model(model, val_set, best_hp_comb)

    return best_hp_comb

def final_test(fp_in, fp_out, hp_comb, iterations, verbose=True, 
               pretrained=False):
    
    '''Retrains the model on the test set multiple times, producing graphical 
       summaries for the first iteration as well as some graphical summaries 
       of the distribution of predictions
       
       Args:
         fp_in: filepath to the folder with the test indexes and sets
         fp_out: filepath to the csv file that stores the results
          - appends results to the csv so that multiple runs are stored in the 
            same csv file
         hp_comb: dictionary of hyperparameters
         iterations: number of times to retrain the model
         verbose: whether to print written outputs and progress to console
         pretrained: whether to use the predictions already stored in the csv 
                     file. Will not retrain the model if True.'''

    train_set = ClaimsDataset(hp_comb['target_col'], 
                              fp_in + 'train_index.csv', 
                              fp_in + 'train_set.csv', 
                              hp_comb['version_no'],
                              hp_comb['include_incurreds'])

    val_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_in + 'val_index.csv', 
                            fp_in + 'val_set.csv', 
                            hp_comb['version_no'],
                            hp_comb['include_incurreds'])

    test_set = ClaimsDataset(hp_comb['target_col'], 
                             fp_in + 'test_index.csv', 
                             fp_in + 'test_set.csv', 
                             hp_comb['version_no'],
                             hp_comb['include_incurreds'])

    preds_matrix = np.zeros(shape=(iterations, len(test_set.index.index)))
    vsInc_list = np.zeros(shape=iterations)
    weighted_vsInc_list = np.zeros(shape=iterations)

    if pretrained == False:
        for i in range(iterations):
            print(f'Iteration {i}')

            model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                            hp_comb['nOut'], hp_comb['version_no'], 
                            hp_comb['type'], hp_comb['nonlinearity'], 
                            hp_comb['output_layer'], hp_comb['dropout'], 
                            hp_comb['normalisation'], hp_comb['include_incurreds']).to(device)
            
            optimiser = optim.Adam(model.parameters(), lr=hp_comb['lr'])

            train_network(model, train_set, hp_comb, optimiser, verbose, val_set)
            
            if verbose:
                print('Test:')
            
            actuals, preds, incurreds = get_preds_actuals(model, test_set, 
                                                          hp_comb, verbose)

            preds_matrix[i] = preds
            preds.to_frame().T.to_csv(fp_out, mode='a', header=False, 
                                      index=False)
            
            vsInc_list[i] = get_vsInc(actuals, preds, incurreds)
            weighted_vsInc_list[i] = get_weighted_vsInc(actuals, preds, 
                                                        incurreds)

            # Produce summaries for the first iteration
            if i == 0:
                analyse_model(model, test_set, hp_comb)

    # Skips training and uses the predictions already stored in the csv file
    elif pretrained == True:
        preds_matrix = pd.read_csv(fp_out, header=None).to_numpy()
        actuals = test_set.index["claim_size"]
        incurreds = test_set.index["latest_incurred"]

        vsInc_list = np.array([get_vsInc(actuals, preds, incurreds) 
                               for preds in preds_matrix])
        
        weighted_vsInc_list = np.array([get_weighted_vsInc(actuals, 
                                                           preds, 
                                                           incurreds) 
                                        for preds in preds_matrix])

    else:
        raise ValueError("pretrained must be True or False")

    # Assessing distribution of aggregate claims
    aggregate_preds = preds_matrix.sum(axis=1)
    aggregate_actuals = actuals.sum()
    aggregate_incurreds = incurreds.sum()

    val_date = 40

    # finding aggregate incurred at the valuation date (calendar quarter 40)
    preds_val = preds_matrix[:, test_set.index['pred_time'] == val_date]
    actuals_val = actuals[test_set.index['pred_time'] == val_date]
    incurreds_val = incurreds[test_set.index['pred_time'] == val_date]

    aggregate_preds_val = preds_val.sum(axis=1)
    aggregate_actuals_val = actuals_val.sum()
    aggregate_incurreds_val = incurreds_val.sum()

    # finding total paid and ocl at the valuation date
    test_paid = test_set.set.loc[test_set.set['pred_time'] == val_date, 
                                 ['claim_no', 'paid']].groupby(['claim_no']).max()
    
    val_date_paid = test_paid.sum()['paid']

    ocl_preds = aggregate_preds_val - [val_date_paid] * len(aggregate_preds_val)
    ocl_actuals = aggregate_actuals_val - val_date_paid
    ocl_incurreds = aggregate_incurreds_val - val_date_paid

    # weighted vsInc at the valuation date
    weighted_vsInc_list_val = np.array([get_weighted_vsInc(actuals_val, 
                                                           preds, 
                                                           incurreds_val)
                                        for preds in preds_val])


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

    # Histogram of distribution of weighted vsInc
    plt.hist(weighted_vsInc_list, weights=(np.zeros_like(weighted_vsInc_list) + 
                                           1. / weighted_vsInc_list.size), 
                                           color='lightgreen')
    
    plt.xlabel('weighted vsInc (%)')
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

    # Histogram of weighted vsInc at the valuation date
    plt.hist(weighted_vsInc_list_val, 
             weights=(np.zeros_like(weighted_vsInc_list_val) + 
                      1. / weighted_vsInc_list_val.size), 
             color='lightgreen')
    
    plt.xlabel('weighted vsInc (%) at valuation date')
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

def create_grid(version_nos, target_cols, criterions, types, output_layers, 
                nOuts, epochss, nHiddens, nLayerss, patiences, batch_sizes, 
                lrs, nonlinearitys, dropouts, normalisations, include_incurredss):
    
    '''
    Inputs: lists of hyperparameter values
    Output: a list of dictionaries to be used in the cross_validate function'''

    hyperparameter_grid = []

    for params in product(version_nos, target_cols, criterions, types, 
                          output_layers, nOuts, epochss, nHiddens, nLayerss, 
                          patiences, batch_sizes, lrs, nonlinearitys, 
                          dropouts, normalisations, include_incurredss):
        
        hyperparameter_grid.append({
            'version_no': params[0],
            'target_col': params[1],
            'criterion': params[2],
            'type': params[3],
            'output_layer': params[4],
            'nOut': params[5],
            'epochs': params[6],
            'nHidden': params[7],
            'nLayers': params[8],
            'patience': params[9],
            'batch_size': params[10],
            'lr': params[11],
            'nonlinearity': params[12],
            'dropout': params[13],
            'normalisation': params[14],
            'include_incurreds': params[15]
        })

    return hyperparameter_grid