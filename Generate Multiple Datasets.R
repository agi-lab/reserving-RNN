# Simulates max_iter number of datasets from the default SynthETIC/SPLICE setup
# Currently saves datasets without (regular) inflation, but can change to inflated datasets

setwd('C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model')

library(SPLICE)
library(ChainLadder)
source("Generate Dataset.R")

fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
max_iter <- 20
seed_base <- 2024
with_cov = TRUE
for_aggregate = FALSE

### Data Generation ############################################################
for (iter in 1:max_iter) {
  generate_dataset(seed_base + iter, with_cov, fp, for_aggregate)
}


# For the same datasets that are simulated above, the below code estimates the
# OCLs using aggregate models such as paid and incurred chain ladders, MCL, and
# the paid-incurred chain model


### Aggregate Models ###########################################################
actual_OCLs <- rep(0, max_iter)
mcl_paid_OCLs <- rep(0, max_iter)
mcl_incurred_OCLs <- rep(0, max_iter)
latest_OCLs <- rep(0, max_iter)
cl_paid_OCLs <- rep(0, max_iter)
cl_incurred_OCLs <- rep(0, max_iter)
pic_OCLs <- rep(0, max_iter)

for_aggregate = TRUE

for (iter in 1:max_iter) {
  seed = seed_base + iter
  set.seed(seed)

  aggregate_data <- generate_dataset(seed, with_cov, fp, for_aggregate)
  
  # Converting individual data to aggregate
  
  incurred_cumulative_triangle = output_incurred(aggregate_data$test, 
                                                 aggregate_level = 2,
                                                 incremental = F, 
                                                 future = F)
  
  payments_cumulative_triangle = claim_output(aggregate_data$test_claims$frequency_vector, 
                                              aggregate_data$test_claims$payment_time_list, 
                                              aggregate_data$test_claims$payment_size_list, 
                                              aggregate_level = 2, 
                                              incremental = F, 
                                              future = F)
  
  #incurred_cumulative_square = output_incurred(aggregate_data$test, 
  #                                             aggregate_level = 2, 
  #                                             incremental = F, 
  #                                             future = T)
  
  payments_cumulative_square = claim_output(aggregate_data$test_claims$frequency_vector, 
                                            aggregate_data$test_claims$payment_time_list, 
                                            aggregate_data$test_claims$payment_size_list, 
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
