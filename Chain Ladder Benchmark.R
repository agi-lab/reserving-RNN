# Simulates max_iter number of datasets from the default SynthETIC/SPLICE setup
# Currently saves datasets without (regular) inflation, but can change to inflated datasets

setwd('C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model')

library(SPLICE)
library(ChainLadder)
library(data.table)

chain_ladder_error = function(fp, seed) {
  # uses the chain ladder to predict the OCL at the valuation date, 
  # then strips out the IBNR components and compares to the true OCL (with IBNR stripped out)
  
  val_date = 40
  
  fp_aggregate <- paste0(fp, "triangle_noInf_cov_TRUE_seed_", seed, ".csv")
  fp_individual <- paste0(fp, "data_noInf_cov_TRUE_seed_", seed, ".csv")
  
  cumulative_payments_triangle <- read.csv(fp_aggregate, row.names = 1)
  individual_claims <- fread(fp_individual)
  individual_claims[, V1 := NULL]
  
  IBNR_total_claims <- individual_claims[txn_delay == 0 & txn_time >= val_date, sum(claim_size)]
  total_claims <- individual_claims[txn_delay == 0, sum(claim_size)]
  
  val_date_paid <- 0
  for (AP in 1:nrow(cumulative_payments_triangle)) {
    val_date_paid = val_date_paid + cumulative_payments_triangle[AP, nrow(cumulative_payments_triangle) - AP + 1]
  }
  
  true_ocl <- total_claims - val_date_paid - IBNR_total_claims
  print(paste0("true ocl excluding IBNR: ", true_ocl))
  print(paste0("total claims: ", total_claims))
  print(paste0("IBNR: ", IBNR_total_claims))
  print(paste0("true ocl including IBNR: ", total_claims - val_date_paid))
  
  
  # Paid Chain Ladder
  cl_paid = chainladder(cumulative_payments_triangle)
  cl_ultimate_paid = sum(predict(cl_paid)[, nrow(cumulative_payments_triangle)])
  
  cl_ocl <- cl_ultimate_paid - val_date_paid - IBNR_total_claims
  
  cl_error = (cl_ocl - true_ocl) / true_ocl
  
  return(cl_error)
}


# this dataset is not complexity 5, maybe complexity 1?
seed = 20201006
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
chain_ladder_error(fp, seed)


# need to beat 26% (if annual chain ladder) or 20% (quarterly chain ladder) as a benchmark reserve error
seed = 20250101
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
chain_ladder_error(fp, seed)

# same even on a larger dataset
seed = as.integer(300000)
fp <- 'C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model/Datasets/R Outputs/'
chain_ladder_error(fp, seed)




