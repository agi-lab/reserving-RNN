###    THIS FILE IS DESIGNED TO EITHER BE RUN ON ITS OWN (WILL HAVE TO    ###
###          UNCOMMENT THE FINAL LINE) OR TO BE IMPORTED INTO             ###
###                   'Generate Multiple Datasets.R'                      ###


library(SPLICE)
library(data.table)

generate_dataset <- function(seed, with_cov, fp, for_aggregate, exposure) {
  # 'with_cov' is a boolean for whether to simulate covariates (True) or not (False)
  
  # Should generate a dataset with and without covariates under the same seed to 
  # make sure that the covariates are actually affecting the simulated 
  # claim sizes, payments and revisions.
  
  # When using the dataset for modelling, it should be simulated with covariate 
  # data (even if the model does not use the covariate data)
  
  # 'for_aggregate' is a boolean specifying whether the generated data is to be 
  # used for an aggregate model (e.g. Chain Ladder) or for an individual model (e.g. our RNN)
  
  set.seed(seed)

  # Claim Occurrence (Default exposure is 12k, so our frequency should be around 
  # 67% higher than default)
  # for 'small' datasets we use 20k as exposure, for 'large' datasets we use 100k (only using 'small' for now)
  n_vector <- claim_frequency(E = exposure, freq = 0.03)
  occurrence_times <- claim_occurrence(frequency_vector = n_vector)
  
  # Claim Size
  claim_sizes <- claim_size(frequency_vector = n_vector)
  
  if (with_cov) {
    claim_sizes_covariates_adj <- claim_size_adj(test_covariates_obj, claim_sizes)
    claim_sizes <- claim_sizes_covariates_adj$claim_size_adj
    covariates_data_obj <- claim_sizes_covariates_adj$covariates_data
  }
  
  # Claim Notification
  notidel <- claim_notification(n_vector, claim_sizes)
  
  # Claim Closure
  setldel <- claim_closure(n_vector, claim_sizes)
  
  # Partial Payments
  no_payments <- claim_payment_no(n_vector, claim_sizes)
  payment_sizes <- claim_payment_size(n_vector, claim_sizes, no_payments)
  
  # Claim Payment Time
  payment_delays <- claim_payment_delay(n_vector, claim_sizes, 
                                        no_payments, setldel)
  
  payment_times <- claim_payment_time(n_vector, occurrence_times, 
                                      notidel, payment_delays)
  
  # Claim Inflation
  payment_inflated <- claim_payment_inflation(n_vector, payment_sizes,
                                              payment_times, occurrence_times,
                                              claim_sizes)
  
  test_claims <- claims(n_vector, occurrence_times, claim_sizes, notidel, 
                        setldel, no_payments, payment_sizes, payment_delays, 
                        payment_times, payment_inflated)
  
  
  # major revisions
  major <- claim_majRev_freq(test_claims)
  major <- claim_majRev_time(test_claims, major)
  major <- claim_majRev_size(major)
  
  # minor revisions
  minor <- claim_minRev_freq(test_claims)
  minor <- claim_minRev_time(test_claims, minor)
  minor <- claim_minRev_size(test_claims, major, minor)
  
  # development of case estimates
  test <- claim_history(test_claims, major, minor)
  
  # Output data for triangles
  if (for_aggregate) {
    aggregate_data <- list(test = test, test_claims = test_claims)
    return(aggregate_data)
    
  }
  
  # Otherwise output data for individual models
  
  # transactional data
  test_incurred_dataset_noInf <- generate_incurred_dataset(test_claims, test)
  
  test_incurred_dataset_noInf <- data.table(test_incurred_dataset_noInf)
  
  
  # append covariate values to each row in the main dataset
  if (with_cov) {
    covariates_features <- data.table(covariates_data_obj$data)
    
    nrows = as.vector(table(test_incurred_dataset_noInf[, claim_no]))
    covariates_features[, nrows := nrows]

    test_incurred_dataset_noInf <- cbind(test_incurred_dataset_noInf, 
                                         covariates_features[rep(1:.N, nrows)])
    
    # remove unnecessary columns
    test_incurred_dataset_noInf[, c("multiplier", "nrows") := NULL]
  }
  
  # rounding values to nearest dollar
  # note that after rounding, it is not guaranteed for cumpaid + OCL = incurred
  # 0.5 is included to round to the nearest integer instead of always rounding down
  test_incurred_dataset_noInf[, ':=' (claim_size = as.integer(0.5 + claim_size), 
                                      incurred = as.integer(0.5 + incurred), 
                                      OCL = as.integer(0.5 + OCL), 
                                      cumpaid = as.integer(0.5 + cumpaid))]
  
  
  # adding accident quarter data
  occurrence_times_per_claim <- c(occurrence_times, recursive=T)
  occurrence_times_per_claim <- ceiling(occurrence_times_per_claim)
  
  
  for (claimno in 1:test_incurred_dataset_noInf[, max(claim_no)]) {
    test_incurred_dataset_noInf[claim_no == claimno, 
                                acc_quarter := occurrence_times_per_claim[claimno]]
    
  }
  
  write.csv(test_incurred_dataset_noInf, paste0(fp, 'data_noInf_cov_', with_cov, 
                                                '_seed_', as.integer(seed), '.csv'))
  
}

################################################################################

seed <- 20201006
with_cov = TRUE
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
for_aggregate <- FALSE
exposure <- 20000

generate_dataset(seed, with_cov, fp, for_aggregate, exposure)


seed <- 20250101
with_cov = TRUE
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
for_aggregate <- FALSE
exposure <- 100000

generate_dataset(seed, with_cov, fp, for_aggregate, exposure)



seed <- 300000
with_cov = TRUE
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
for_aggregate <- FALSE
exposure <- 300000

generate_dataset(seed, with_cov, fp, for_aggregate, exposure)




################################################################################

# much shorter and simpler alternative to above code
for (complexity in 1:5) {
  # 20,000 epy * 3% frequency / 4 quarters
  test_dataset <- generate_data(n_claims_per_period = 20000 * 0.03 / 4,
                n_periods = 40,
                complexity = complexity,
                random_seed = 1,
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
  
  write.csv(test_incurred_dataset, paste0(fp, 'complexity ', complexity, '.csv'))
  
}

