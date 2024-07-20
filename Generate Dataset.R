###    THIS FILE IS DESIGNED TO EITHER BE RUN ON ITS OWN (WILL HAVE TO    ###
###          UNCOMMENT THE FINAL LINE) OR TO BE IMPORTED INTO             ###
###                   'Generate Multiple Datasets.R'                      ###


library(SPLICE)

seed <- 20201006
with_cov = TRUE
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'

generate_dataset <- function(seed, with_cov, fp) {
  # with_cov is a boolean for whether to simulate covariates (True) or not (False)
  
  set.seed(seed)

  # Claim Occurrence (claim numbers increased from default by a factor of 3)
  n_vector <- claim_frequency(E = 36000, freq = 0.03)
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
  
  
  # transactional data
  test_incurred_dataset_noInf <- generate_incurred_dataset(test_claims, test)
  
  write.csv(test_incurred_dataset_noInf, paste0(fp, 'data_noInf_cov_', with_cov, 
                                                '_seed_', seed, '.csv'))
  
  if (with_cov) {
    write.csv(covariates_data_obj$data, paste0(fp, 'covariate_data_seed_',
                                               seed, '.csv'))
  }
  
}

generate_dataset(seed, with_cov, fp)