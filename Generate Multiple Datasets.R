# Simulates max_iter number of datasets from the default SynthETIC/SPLICE setup
# Currently saves datasets without (regular) inflation, but can change to inflated datasets

setwd('C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model')

library(SPLICE)
library(ChainLadder)
source("Generate Dataset.R")

fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
max_iter <- 10
seed_base <- 2024
with_cov = TRUE

# Data Generation
for (iter in 1:max_iter) {
  generate_dataset(seed_base + iter, with_cov, fp)
}






# MCL and CL
actual_OCLs <- rep(0, max_iter)
mcl_paid_OCLs <- rep(0, max_iter)
mcl_incurred_OCLs <- rep(0, max_iter)
latest_OCLs <- rep(0, max_iter)
cl_paid_OCLs <- rep(0, max_iter)
cl_incurred_OCLs <- rep(0, max_iter)
pic_OCLs <- rep(0, max_iter)

for (iter in 1:max_iter) {
  set.seed(seed_base + iter)
  
  # Claim Occurrence
  n_vector <- claim_frequency(E = 36000, freq = 0.03)
  occurrence_times <- claim_occurrence(frequency_vector = n_vector)
  
  # Claim Size
  claim_sizes <- claim_size(frequency_vector = n_vector)
  
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
  
  # Converting individual data to aggregate
  
  incurred_cumulative_triangle = output_incurred(test, 
                                                 aggregate_level = 2,
                                                 incremental = F, 
                                                 future = F)
  
  payments_cumulative_triangle = claim_output(test_claims$frequency_vector, 
                                              test_claims$payment_time_list, 
                                              test_claims$payment_size_list, 
                                              aggregate_level = 2, 
                                              incremental = F, 
                                              future = F)
  
  #incurred_cumulative_square = output_incurred(test, 
  #                                             aggregate_level = 2, 
  #                                             incremental = F, 
  #                                             future = T)
  
  payments_cumulative_square = claim_output(test_claims$frequency_vector, 
                                            test_claims$payment_time_list, 
                                            test_claims$payment_size_list, 
                                            aggregate_level = 2, 
                                            incremental = F, 
                                            future = T)
  
  # Munich Chain Ladder
  
  mcl = MunichChainLadder(payments_cumulative_triangle, 
                          incurred_cumulative_triangle, 
                          tailP = F, tailI = F)
  
  mcl_ultimate_paid = summary(mcl)$Totals$Paid[2]
  mcl_ultimate_incurred = summary(mcl)$Totals$Incurred[2]
  
  actual_ultimate_paid = sum(payments_cumulative_square[, 20])
  
  latest_paid = summary(mcl)$Totals$Paid[1]
  latest_incurred = summary(mcl)$Totals$Incurred[1]
  
  actual_OCLs[iter] = actual_ultimate_paid - latest_paid
  mcl_paid_OCLs[iter] = mcl_ultimate_paid - latest_paid
  mcl_incurred_OCLs[iter] = mcl_ultimate_incurred - latest_paid
  latest_OCLs[iter] = latest_incurred - latest_paid
  
  # Paid Chain Ladder
  cl_paid = chainladder(payments_cumulative_triangle)
  cl_ultimate_paid = sum(predict(cl_paid)[, 20])
  cl_paid_OCLs[iter] = cl_ultimate_paid - latest_paid
  
  # Incurred Chain Ladder
  cl_incurred = chainladder(incurred_cumulative_triangle)
  cl_ultimate_incurred = sum(predict(cl_incurred)[, 20])
  cl_incurred_OCLs[iter] = cl_ultimate_incurred - latest_paid
  
  # Paid-Incurred Chain
  pic = PaidIncurredChain(payments_cumulative_triangle, incurred_cumulative_triangle)
  pic_ultimate = pic$Ult.Loss
  pic_OCLs[iter] = pic_ultimate - latest_paid
  
  
}

summary(actual_OCLs)
summary(mcl_paid_OCLs)
summary(mcl_incurred_OCLs)
summary(latest_OCLs)
summary(cl_paid_OCLs)
summary(cl_incurred_OCLs)
summary(pic_OCLs)

mcl_paid_OCL_error <- abs(mcl_paid_OCLs - actual_OCLs) / actual_OCLs
mcl_incurred_OCL_error <- abs(mcl_incurred_OCLs - actual_OCLs) / actual_OCLs
latest_OCL_error <- abs(latest_OCLs - actual_OCLs) / actual_OCLs
cl_paid_OCL_error <- abs(cl_paid_OCLs - actual_OCLs) / actual_OCLs
cl_incurred_OCL_error <- abs(cl_incurred_OCLs - actual_OCLs) / actual_OCLs
pic_OCL_error <- abs(pic_OCLs - actual_OCLs) / actual_OCLs

summary(mcl_paid_OCL_error)
summary(mcl_incurred_OCL_error)
summary(latest_OCL_error)
summary(cl_paid_OCL_error)
summary(cl_incurred_OCL_error)
summary(pic_OCL_error)
