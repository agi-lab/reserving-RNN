### NOTES #####################################################################

#Uses environment 'Reserving-RNN-PyTorch'
#python 3.10.13
#pytorch 2.2.1
#torchvision 0.17.1
#cuda 12.1.0
#pandas 2.1.1
#numpy 1.26.0

# Works like any of the 'Model Training Triangular' files but applies to 
# multiple simulated datasets

# Stores outputs from all datasets in a csv file

### PACKAGES ##################################################################

from Functions import *

### VARIABLES (ADJUST THESE BEFORE RUNNNING) ##################################

max_iter = 20
seed_base = 2024

#fp_py = 'Datasets/Python Inputs/noInf_WithInc_seed_'
#fp_out = 'Results/Results Multiple Datasets WithInc WithCov.csv'

fp_py = 'Datasets/Python Inputs/noInf_NoInc_seed_'
fp_out = 'Results/Results Multiple Datasets NoInc NoCov.csv'

# Model to test
target_cols = ['log_m']
criterions = [torch.nn.L1Loss()]
types = ['LSTM']
output_layers = ['linear']
nOuts = [1]
epochss = [200]
nonlinearitys = ['relu']
patiences = [10]
normalisations = [True]
include_incurredss = [False]
include_covariatess = [False]
transform_inputss = [True]

nHiddens = [512]
nLayerss = [2]
batch_sizes = [512]
dropouts = [0.2]
learning_rates = [0.001]


hyperparameter_grid = create_grid(target_cols, criterions, types, output_layers, 
                                  nOuts, epochss, nHiddens, nLayerss, patiences, 
                                  batch_sizes, learning_rates, nonlinearitys, 
                                  dropouts, normalisations, include_incurredss, 
                                  include_covariatess, transform_inputss)

hp_comb = hyperparameter_grid[0]

### MODEL TESTING #############################################################

print('MODEL TESTING')

for i in range(1, max_iter + 1):
    torch.manual_seed(1)
    fp_py_full = fp_py + str(i + seed_base) + '/'

    print('Seed: ' + str(i + seed_base))

    train_set = ClaimsDataset(hp_comb['target_col'], 
                              fp_py_full + 'train_index.csv', 
                              fp_py_full + 'train_set.csv', 
                              hp_comb['include_incurreds'],
                              hp_comb['include_covariates'],
                              hp_comb['transform_inputs'])

    val_set = ClaimsDataset(hp_comb['target_col'], 
                            fp_py_full + 'val_index.csv', 
                            fp_py_full + 'val_set.csv', 
                            hp_comb['include_incurreds'],
                            hp_comb['include_covariates'],
                            hp_comb['transform_inputs'])

    test_set = ClaimsDataset(hp_comb['target_col'], 
                             fp_py_full + 'test_index.csv', 
                             fp_py_full + 'test_set.csv', 
                             hp_comb['include_incurreds'],
                             hp_comb['include_covariates'],
                             hp_comb['transform_inputs'])


    model = ClaimsRNN(hp_comb['nHidden'], hp_comb['nLayers'], 
                            hp_comb['nOut'], hp_comb['type'], hp_comb['nonlinearity'], 
                            hp_comb['output_layer'], hp_comb['dropout'], 
                            hp_comb['normalisation'], hp_comb['include_incurreds'], 
                            hp_comb['include_covariates']).to(device)
            
    optimiser = optim.Adam(model.parameters(), lr=hp_comb['lr'])

    train_network(model, train_set, hp_comb, optimiser, True, val_set)
            
    print('Test:')
            
    actuals, preds, incurreds = get_preds_actuals(model, test_set, hp_comb, True)
            
    #weighted_vsInc = get_weighted_vsInc(actuals, preds, incurreds)

    val_date = 40

    # finding aggregate incurred at the valuation date (calendar quarter 40)
    preds_val = preds[test_set.index['pred_time'] == val_date]
    actuals_val = actuals[test_set.index['pred_time'] == val_date]
    incurreds_val = incurreds[test_set.index['pred_time'] == val_date]

    aggregate_preds_val = preds_val.sum()
    aggregate_actuals_val = actuals_val.sum()
    aggregate_incurreds_val = incurreds_val.sum()

    # finding total paid and ocl at the valuation date
    test_paid = test_set.set.loc[test_set.set['pred_time'] == val_date, 
                                 ['claim_no', 'paid']].groupby(['claim_no']).max()
    
    val_date_paid = test_paid.sum()['paid']

    ocl_preds = aggregate_preds_val - val_date_paid
    ocl_actuals = aggregate_actuals_val - val_date_paid
    ocl_incurreds = aggregate_incurreds_val - val_date_paid

    ocl_error_preds = round_threshold(abs(ocl_preds - ocl_actuals) / ocl_actuals)
    ocl_error_incurreds = round_threshold(abs(ocl_incurreds - ocl_actuals) / ocl_actuals)

    # weighted vsInc at the valuation date
    weighted_vsInc_val = round_threshold(get_weighted_vsInc(actuals_val, preds_val, incurreds_val))

    results = pd.DataFrame({'Seed': [i + seed_base], 
                            'ocl_actuals': [ocl_actuals], 
                            'ocl_preds': [ocl_preds], 
                            'ocl_incurreds': [ocl_incurreds], 
                            'ocl_error_preds': [ocl_error_preds], 
                            'ocl_error_incurreds': [ocl_error_incurreds], 
                            'weighted_vsInc_val': [weighted_vsInc_val]})
    
    results.to_csv(fp_out, mode = 'a', header = True, index = False)


print('MODEL TESTING COMPLETE')
