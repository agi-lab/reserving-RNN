library(data.table)

# Getting the path of current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
print( getwd() )

distort_case_estimates <- function(fp_in, fp_out, prop_low, prop_high, seed) {
  
  set.seed(seed)
  
  data <- fread(fp_in)
  
  # Find end time for each claim
  data[, duration := max(txn_delay), keyby = claim_no]
  
  # Factor for scaling each claim
  revision_factors = runif(data[, length(unique(claim_no))], prop_low, prop_high)
  claim_numbers = 1:data[, length(unique(claim_no))]
  revisions_by_claim = data.table(claim_no = claim_numbers, revision_factor = revision_factors)
  data = merge(data, revisions_by_claim, by = "claim_no")
  
  # 'simple' version
  data[, OCL := round(OCL * (1 - revision_factor * (duration - txn_delay) / duration))]
  data[, incurred := OCL + cumpaid]

  # # Where there is an existing case estimate revision,
  # # scale case estimates down by proportion of time remaining
  # # and round them to nearest dollar
  # data[, ':=' (orig_incurred = incurred,
  #              orig_OCL = OCL)]
  # 
  # data[txn_type != "P", OCL := round(OCL * (1 - revision_factor * (duration - txn_delay) / duration))]
  # data[, incurred := OCL + cumpaid]
  # 
  # # The final revision will need to be reverted to the true incurred so that case estimates continue to converge to 0
  # data[, last_revision := FALSE]
  # data[txn_type != "P", last_revision := seq_len(.N) == .N, by = claim_no]
  # 
  # data[last_revision == TRUE, incurred := orig_incurred]
  # data[last_revision == TRUE, OCL := incurred - cumpaid]
  # 
  # # Where there is only a payment,
  # # copy the incurred from the previous transaction and update OCL accordingly
  # data[txn_type == "P", incurred := NA]
  # data[, incurred := nafill(incurred, type = "locf")]
  # data[txn_type == "P", OCL := incurred - cumpaid]


  data[, ':=' (duration = NULL,
              revision_factor = NULL)]

  fwrite(data, fp_out)
}


prop_low = 0.7
prop_high = 0.9
fp_in_base = './Datasets/R Outputs/data_noInf_cov_TRUE_seed_'
fp_out_base = "./Datasets/R Outputs/Distorted Case Estimates/data_noInf_cov_TRUE_seed_"

seed_base = 500
max_iter = 50

for (i in 0:max_iter) {
  fp_in = paste0(fp_in_base, seed_base + i, '.csv') 
  fp_out = paste0(fp_out_base, seed_base + i, '.csv')
  
  print(paste0('Seed: ', i + seed_base))
  #print(fp_in)
  distort_case_estimates(fp_in, fp_out, prop_low, prop_high, seed_base + i)
}
print("Done!")

