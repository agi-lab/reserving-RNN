library(data.table)

# Getting the path of current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
print( getwd() )


noisy_case_estimates <- function(fp_in, fp_out, prop_low, prop_high, seed) {
  
  set.seed(seed)

  data <- fread(fp_in)
  
  # revisions <- data[, diff(incurred)]
  # revisions <- c(0, revisions)
  # 
  # data[, revision := revisions]
  # data[txn_delay == 0, revision := incurred]
  # data[, ':=' (orig_incurred = incurred,
  #              orig_OCL = OCL)]
  
  # 1. Add noise to case estimates at revision times
  revision_factors = runif(data[txn_type != "P", .N], min = 0.7, max = 1.3)
  data[txn_type != "P", revision_factor := revision_factors]
  
  data[, OCL := round(OCL * revision_factor)]
  data[cumpaid == incurred & is.na(OCL), OCL := 0]
  data[, incurred := cumpaid + OCL]
  
  data[, OCL_min := round(OCL * 0.1)]
  
  # data[txn_delay == 0 & is.na(OCL), .N]
  # data[cumpaid == incurred & is.na(OCL), .N]
  # data[is.na(OCL), .N]
  # data[is.na(OCL) & txn_type != "P", .N]
  
  # 2. Roll noise adjustment forward
  data[, incurred := nafill(incurred, type = 'locf')]
  data[, OCL := incurred - cumpaid]
  
  data[, OCL_min := nafill(OCL_min, type = 'locf')]
  
  # 3. Correct negative case estimates to be 10% of previous case estimate
  # data[OCL < 0, incurred := incurred - OCL]
  # data[, OCL := incurred - cumpaid]
  
  data[OCL < 0, OCL := OCL_min]
  data[, incurred := OCL + cumpaid]
  
  # data[OCL < 0, .N]
  
  data[, revision_factor := NULL]
  data[, OCL_min := NULL]
  

  fwrite(data, fp_out)
}

fp_in = './Datasets/R Outputs/data_noInf_cov_TRUE_seed_500.csv'
seed = 500

prop_low = 0.7
prop_high = 1.3
fp_in_base = './Datasets/R Outputs/data_noInf_cov_TRUE_seed_'
fp_out_base = "./Datasets/R Outputs/Noisy Case Estimates/data_noInf_cov_TRUE_seed_"

seed_base = 500
max_iter = 50

for (i in 0:max_iter) {
  fp_in = paste0(fp_in_base, seed_base + i, '.csv') 
  fp_out = paste0(fp_out_base, seed_base + i, '.csv')
  
  print(paste0('Seed: ', i + seed_base))
  #print(fp_in)
  noisy_case_estimates(fp_in, fp_out, prop_low, prop_high, seed_base + i)
}
print("Done!")









revisions <- data[, diff(incurred)]
revisions <- c(0, revisions)

data[, revision := revisions]
data[txn_delay == 0, revision := incurred]

# 1. figure out maximum allowable reduction in OCL
# i.e. figure out how many payments are made before the next revision
data[revision != 0, ]
rows_with_payments <- data[txn_type == "P" & (shift(txn_type, type = "lead") != "P" | is.na(shift(txn_type, type = "lead"))), which = TRUE]

data[rows_with_payments, max_OCL_reduction := orig_OCL]

data[, max_OCL_reduction := nafill(max_OCL_reduction, type = 'nocb')]
data[txn_type != "P" & OCL != 0, max_OCL_reduction_ratio := pmin(orig_OCL, max_OCL_reduction) / orig_OCL]

# fix max revision factor for rows with multiple revisions consecutively
rows_with_consecutive_revisions <- data[txn_type != "P" & shift(txn_type, type = "lead") != "P", which = TRUE]
data[rows_with_consecutive_revisions, max_OCL_reduction_ratio := 1]

# 2. multiply by factor (random either size of 1)
# make sure you do not adjust raw revision as it won't align anymore

# data[, .(1 - pmin(0.3, max_OCL_reduction_ratio), (1 + pmin(0.3, max_OCL_reduction_ratio)))]

data[, revision_factor := mapply(runif, n = 1, min = 1 - pmin(0.3, max_OCL_reduction_ratio), max = (1 + pmin(0.3, max_OCL_reduction_ratio)))]
data[, additional_revision := OCL * (1 - revision_factor)]
data[, additional_revision := nafill(additional_revision, type = 'locf')]
data[, OCL := OCL + additional_revision]

data[, incurred := OCL + cumpaid]
total_revisions <- data[, diff(incurred)]
total_revisions <- c(0, total_revisions)

data[, total_revision := total_revisions]

data[OCL < 0, ]

View(data[claim_no == 29864])


# data[, summary(max_OCL_reduction_ratio)]
# data[max_OCL_reduction_ratio == 1, .N]
# data[max_OCL_reduction_ratio > 0 & max_OCL_reduction_ratio < 1, .N]
# data[max_OCL_reduction_ratio < 0.3, .N]
# data[max_OCL_reduction_ratio < 0.1 & max_OCL_reduction_ratio > 0, summary(max_OCL_reduction_ratio)]


data[txn_type != 'P', summary(revision_factor)]
data[txn_type != 'P', summary(revision)]
data[txn_type != 'P', summary(additional_revision)]
data[txn_type != 'P', summary(total_revision)]



data[txn_type != 'P', hist(revision_factor)]
data[txn_type != 'P', hist(additional_revision)]
data[txn_type != 'P', hist(revision)]


View(data[additional_revision == min(additional_revision)])
View(data[additional_revision == max(additional_revision)])


data[, hist(OCL)]
data[, hist(orig_OCL)]

data[, summary(OCL)]
data[, summary(orig_OCL)]
data[, summary(incurred)]
data[, summary(claim_size)]

# proportion of claims that cannot be adjusted at all
data[max_OCL_reduction_ratio == 0 & revision != 0, .N] / data[revision != 0, .N]

# proportion of claims that cannot be adjusted by 0.3
data[max_OCL_reduction_ratio < 0.3 & revision != 0, .N] / data[revision != 0, .N]

data[max_OCL_reduction_ratio < 0.3 & revision != 0 & max_OCL_reduction_ratio != 0, .N] / data[max_OCL_reduction_ratio != 0 & revision != 0, .N]
