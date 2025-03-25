# Takes simulated SPLICE output from R, converts it into a 'set' and 'index' 
# csv file, splits into training and test sets and stores them in csvs

library(data.table)
library(rstudioapi)

# Getting the path of current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
print( getwd() )


################################################################################

data_manipulation <- function(fp_in, fp_out, fp_out_noInc) {

  # Reading data
  data <- fread(fp_in)
  
  # Dropping first column
  data[, V1 := NULL]
  
  
  # Adding duration and start time columns, and rounding them
  data[, ':=' (rept_quarter = ceiling(min(txn_time)),
               finalised_quarter = ceiling(max(txn_time))), keyby=claim_no]
  
  # remove claims that are reported and settled within the same quarter
  data <- data[!(claim_no %in% data[rept_quarter == finalised_quarter, unique(claim_no)])]
  
  # Adding incremental payments to data
  payments <- data[, diff(cumpaid)]
  payments[payments < 0] <- 0
  payments <- c(0, payments)
  
  data[, payment := payments]
  
  
  # Adding incremental case estimate revisions to data
  revisions <- data[, diff(incurred)]
  revisions <- c(0, revisions)
  
  data[, revision := revisions]
  data[txn_delay == 0, revision := incurred]
  
  data[revision == 0, .N]
  data[revision > 0 & txn_delay != 0, .N]
  
  
  # Creating index data
  index_data <- data[, .(pred_time = seq(mean(rept_quarter), mean(finalised_quarter) - 1)), keyby = claim_no]
  
  index_data <- merge(index_data, data[, .(acc_quarter = mean(acc_quarter), 
                                           rept_quarter = mean(rept_quarter), 
                                           finalised_quarter = mean(finalised_quarter), 
                                           claim_size = mean(claim_size), 
                                           `Legal Representation` = unique(`Legal Representation`), 
                                           `Injury Severity` = unique(`Injury Severity`), 
                                           `Age of Claimant` = unique(`Age of Claimant`)), keyby=claim_no], by="claim_no")
  
  # when using nested data table in condition, need column names to be different
  setnames(data, "claim_no", "claim_number")
  
  index_data[, ':=' (incurred = data[claim_number == claim_no & txn_time < pred_time, incurred[length(incurred)]],
                     cumpaid = data[claim_number == claim_no & txn_time < pred_time, cumpaid[length(cumpaid)]]), keyby=.(claim_no, pred_time)]
  
  # having $1 discrepancy is ok due to rounding 
  #View(index_data[incurred != OCL + cumpaid, .(claim_no, incurred, OCL, cumpaid, OCL + cumpaid)])
  
  index_data[, ':=' (index = 0:(.N - 1),
                     dev_quarter = pred_time - rept_quarter + 1,
                     m = claim_size / incurred,
                     log_m = log(claim_size / incurred),
                     true_ocl = claim_size - cumpaid,
                     log_claim_size = log(claim_size),
                     log_true_ocl = log(claim_size - cumpaid))]
  
  
  # Creating databoxes (i.e. 'set' data)
  databoxes <- index_data[, data[claim_number == claim_no & txn_time < pred_time, .(txn_time, txn_delay, cumpaid, OCL, txn_type, payment, revision)], keyby = .(index, claim_no, pred_time)]
  
  
  # renaming columns
  setnames(data, "claim_number", "claim_no")
  setnames(index_data, c("incurred"), c("latest_incurred"))
  setnames(databoxes, c('txn_time', 'txn_delay', 'cumpaid', 'OCL'), c('cal_time', 'dev_time', 'paid', 'ocl'))
  
  # Some claim reports have payments really close to 0, need to set these to exactly 0
  databoxes[paid > 0 & paid < 0.01, paid := 0]
  
  # Binary encode legal rep covariate
  index_data[`Legal Representation` == 'Y', `Legal Representation` := 1]
  index_data[`Legal Representation` == 'N', `Legal Representation` := 0]
  index_data[, `Legal Representation` := as.numeric(`Legal Representation`)]
  
  # Ordinal encode age covariate
  index_data[`Age of Claimant` == '0-15', `Age of Claimant` := 0]
  index_data[`Age of Claimant` == '15-30', `Age of Claimant` := 1]
  index_data[`Age of Claimant` == '30-50', `Age of Claimant` := 2]
  index_data[`Age of Claimant` == '50-65', `Age of Claimant` := 3]
  index_data[`Age of Claimant` == 'over 65', `Age of Claimant` := 4]
  index_data[, `Age of Claimant` := as.numeric(`Age of Claimant`)]
  
  # Make injury severity go from 0-5 instead of 1-6 so it works with embeddings
  index_data[, `Injury Severity` := `Injury Severity` - 1]
  
  # reordering index data columns
  setcolorder(index_data, c('index', 
                            'claim_no', 
                            'pred_time', 
                            'dev_quarter', 
                            'rept_quarter', 
                            'acc_quarter', 
                            'finalised_quarter', 
                            'claim_size', 
                            'log_claim_size', 
                            'latest_incurred', 
                            'm', 
                            'log_m', 
                            'true_ocl', 
                            'log_true_ocl', 
                            'Legal Representation', 
                            'Injury Severity', 
                            'Age of Claimant'))
  
  index_data[, cumpaid := NULL] # can add back later
  
  
  # Adding FNN summary data to index
  payment_summary <- databoxes[payment > 0, .(num_payments = .N,
                               mean_payments = mean(payment),
                               vco_payments = sd(payment) / mean(payment),
                               max_payment = max(payment)), keyby=index]
  
  index_data <- merge(index_data, payment_summary, by='index', all.x=TRUE)
  
  revision_summary <- databoxes[revision != 0 & dev_time > 0, .(num_revisions = .N,
                                               mean_revisions = mean(revision),
                                               max_revision = max(revision),
                                               total_revisions = sum(abs(revision)),
                                               prop_upward_revisions = sum(revision > 0) / .N), keyby=index]
  
  index_data <- merge(index_data, revision_summary, by='index', all.x=TRUE)
  
  index_data[is.na(index_data)] <- 0
  
  ### TRAIN TEST SPLIT ######################################################
  val_start_quarter = 36
  test_start_quarter = 40
  
  # Valuation date is 40
  train_index <- index_data[finalised_quarter <= val_start_quarter]
  val_index <- index_data[finalised_quarter > val_start_quarter & finalised_quarter <= test_start_quarter]
  test_index <- index_data[finalised_quarter > test_start_quarter]
  
  # TESTING: removing all observations in test sets that occur before the final observation in the training set
  val_index <- val_index[pred_time >= val_start_quarter]
  test_index <- test_index[pred_time >= test_start_quarter]
  
  
  # TESTING: randomly splitting by claim (i.e. not by time)
  # train_prop = 0.6
  # val_prop = 0.2
  # test_prop = 0.2
  # 
  # all_claims <- 1:max(index_data$claim_no)
  # 
  # train_claims <- sample(all_claims, train_prop * length(all_claims), replace = FALSE)
  # val_claims <- all_claims[!(all_claims %in% train_claims)]
  # 
  # test_claims <- sample(val_claims, test_prop / (val_prop + test_prop) * length(val_claims), replace = FALSE)
  # val_claims <- val_claims[!(val_claims %in% test_claims)]
  # 
  # train_index <- index_data[claim_no %in% train_claims]
  # val_index <- index_data[claim_no %in% val_claims]
  # test_index <- index_data[claim_no %in% test_claims]
  
  
  
  train_set <- databoxes[index %in% train_index[, index]]
  val_set <- databoxes[index %in% val_index[, index]]
  test_set <- databoxes[index %in% test_index[, index]]
  
  # creating noIncurred versions
  train_set_noInc = train_set[((txn_type != 'Ma') | (paid == 0)) & (txn_type != 'Mi')]
  val_set_noInc = val_set[((txn_type != 'Ma') | (paid == 0)) & (txn_type != 'Mi')]
  test_set_noInc = test_set[((txn_type != 'Ma') | (paid == 0)) & (txn_type != 'Mi')]
  
  # removing unnecessary columns
  train_set[, txn_type := NULL]
  val_set[, txn_type := NULL]
  test_set[, txn_type := NULL]
  
  train_set_noInc[, ':=' (txn_type = NULL, revision = NULL)]
  val_set_noInc[, ':=' (txn_type = NULL, revision = NULL)]
  test_set_noInc[, ':=' (txn_type = NULL, revision = NULL)]
  
  train_index_noInc <- train_index[, .SD, .SDcols = !c('num_revisions', 'mean_revisions', 'max_revision', 'total_revisions', 'prop_upward_revisions')]
  val_index_noInc <- val_index[, .SD, .SDcols = !c('num_revisions', 'mean_revisions', 'max_revision', 'total_revisions', 'prop_upward_revisions')]
  test_index_noInc <- test_index[, .SD, .SDcols = !c('num_revisions', 'mean_revisions', 'max_revision', 'total_revisions', 'prop_upward_revisions')]
  
  
  # Exporting With Incurred (check that not export with row column)
  fwrite(train_index, paste0(fp_out, 'train_index.csv'))
  fwrite(val_index, paste0(fp_out, 'val_index.csv'))
  fwrite(test_index, paste0(fp_out, 'test_index.csv'))
  
  fwrite(train_set, paste0(fp_out, 'train_set.csv'))
  fwrite(val_set, paste0(fp_out, 'val_set.csv'))
  fwrite(test_set, paste0(fp_out, 'test_set.csv'))
  
  # Exporting No Incurred
  fwrite(train_index_noInc, paste0(fp_out_noInc, 'train_index.csv'))
  fwrite(val_index_noInc, paste0(fp_out_noInc, 'val_index.csv'))
  fwrite(test_index_noInc, paste0(fp_out_noInc, 'test_index.csv'))
  
  fwrite(train_set_noInc, paste0(fp_out_noInc, 'train_set.csv'))
  fwrite(val_set_noInc, paste0(fp_out_noInc, 'val_set.csv'))
  fwrite(test_set_noInc, paste0(fp_out_noInc, 'test_set.csv'))
}

################################################################################

# fp_in = './Datasets/R Outputs/small sample.csv'
# fp_out = './Datasets/Python Inputs/small sample WithInc/'
# fp_out_noInc = './Datasets/Python Inputs/small sample NoInc/'


# fp_in = './Datasets/R Outputs/complexity 1.csv'
# fp_out = './Datasets/Python Inputs/complexity 1 WithInc/'
# fp_out_noInc = './Datasets/Python Inputs/complexity 1 NoInc/'

fp_in = './Datasets/R Outputs/data_noInf_cov_TRUE_seed_20201006.csv'
fp_out = './Datasets/Python Inputs/noInf_WithInc_seed_20201006/'
fp_out_noInc = './Datasets/Python Inputs/noInf_NoInc_seed_20201006/'


data_manipulation(fp_in, fp_out, fp_out_noInc)


# creating multiple datasets
max_iter = 20
seed_base = 2024

fp_R = './Datasets/R Outputs/data_noInf_cov_TRUE_seed_'
fp_py_WithInc = './Datasets/Python Inputs/noInf_WithInc_seed_'
fp_py_noInc = './Datasets/Python Inputs/noInf_NoInc_seed_'

for (i in 1:max_iter) {
  fp_in = paste0(fp_R, i + seed_base, '.csv')
  fp_out = paste0(fp_py_WithInc, i + seed_base, '/')
  fp_out_noInc = paste0(fp_py_noInc, i + seed_base, '/')

  print(paste0('Seed: ', i + seed_base))
  #print(paste0('fp_in: ', fp_in))
  #print(paste0('fp out: ', fp_out))

  data_manipulation(fp_in, fp_out, fp_out_noInc)
}
print("Done!")
