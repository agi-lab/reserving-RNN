###    THIS FILE IS DESIGNED TO EITHER BE RUN ON ITS OWN (WILL HAVE TO    ###
###          UNCOMMENT THE FINAL LINE) OR TO BE IMPORTED INTO             ###
###                   'Generate Multiple Datasets.R'                      ###


library(SPLICE)
library(data.table)

################################################################################

generate_dataset_short = function(seed, exposure, complexity) {
  # much shorter and simpler alternative to previous function
  # assumes covariates will always be included
  # complexity is an integer from 1-5 (inclusive) that dictates the complexity of the dataset
  # 1 = least complex, 5 = most complex
  
  # epy * 3% frequency / 4 quarters
  test_dataset <- generate_data(n_claims_per_period = exposure * 0.03 / 4,
                                n_periods = 40,
                                complexity = complexity,
                                random_seed = seed,
                                covariates_obj = test_covariates_obj
  )
  
  occurrence_times <- test_dataset$claim_dataset$occurrence_period
  
  test_incurred_dataset <- data.table(test_dataset$incurred_dataset)
  
  covariates_features <- data.table(test_dataset$covariates_data$data)
  
  nrows = as.vector(table(test_incurred_dataset[, claim_no]))
  covariates_features[, nrows := nrows]
  
  test_incurred_dataset <- cbind(test_incurred_dataset, 
                                 covariates_features[rep(1:.N, nrows)])
  
  # remove unnecessary columns
  test_incurred_dataset[, c("multiplier", "nrows") := NULL]
  
  
  # rounding values to nearest dollar
  # note that after rounding, it is not guaranteed for cumpaid + OCL = incurred
  # 0.5 is included to round to the nearest integer instead of always rounding down
  test_incurred_dataset[, ':=' (claim_size = as.integer(0.5 + claim_size), 
                                incurred = as.integer(0.5 + incurred), 
                                OCL = as.integer(0.5 + OCL), 
                                cumpaid = as.integer(0.5 + cumpaid))]
  
  
  # adding accident quarter data
  occurrence_times_per_claim <- c(occurrence_times, recursive=T)
  occurrence_times_per_claim <- ceiling(occurrence_times_per_claim)
  
  
  for (claimno in 1:test_incurred_dataset[, max(claim_no)]) {
    test_incurred_dataset[claim_no == claimno, 
                          acc_quarter := occurrence_times_per_claim[claimno]]
    
  }
  
  return(test_incurred_dataset)
}

################################################################################

for (complexity in 1:5) {
  test_incurred_dataset <- generate_dataset_short(seed=1, exposure=20000, complexity=complexity)
  write.csv(test_incurred_dataset, paste0(fp, 'complexity ', complexity, '.csv'))
  
}



seed = 200
num_datasets = 100
exposure = 100000
complexity = 2
#fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
fp <- 'C:/Users/matty/Documents/Uni/Unimelb/MCOM/Research Report/Matt Model/Datasets/R Outputs/'


for (iter in 0:num_datasets) {
  print(seed + iter)
  test_incurred_dataset <- generate_dataset_short(seed + iter, exposure, complexity)
  write.csv(test_incurred_dataset, paste0(fp, 'data_noInf_cov_TRUE_seed_', as.integer(seed + iter), '.csv'))
}



seed = 500
num_datasets = 100
exposure = 100000
complexity = 5
#fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
fp <- 'C:/Users/matty/Documents/Uni/Unimelb/MCOM/Research Report/Matt Model/Datasets/R Outputs/'

for (iter in 0:num_datasets) {
  print(seed + iter)
  test_incurred_dataset <- generate_dataset_short(seed + iter, exposure, complexity)
  write.csv(test_incurred_dataset, paste0(fp, 'data_noInf_cov_TRUE_seed_', as.integer(seed + iter), '.csv'))
}

