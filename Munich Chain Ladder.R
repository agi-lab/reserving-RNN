### Loading Packages ###########################################################

library(ChainLadder)
library(SPLICE)

setwd('C:/Users/matty/OneDrive/MCOM/Research Report/Matt Model')

### Reading Data ###############################################################

set.seed(20201006)
test_claims <- SynthETIC::test_claims_object

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


### Converting Individual data to Triangle Data ################################

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

incurred_cumulative_square = output_incurred(test, 
                                             aggregate_level = 2, 
                                             incremental = F, 
                                             future = T)

payments_cumulative_square = claim_output(test_claims$frequency_vector, 
                                          test_claims$payment_time_list, 
                                          test_claims$payment_size_list, 
                                          aggregate_level = 2, 
                                          incremental = F, 
                                          future = T)


### Munich Chain Ladder ########################################################

mcl = MunichChainLadder(payments_cumulative_triangle, 
                        incurred_cumulative_triangle, 
                        tailP = F, tailI = F)

summary(mcl)

mcl_ultimate_paid = summary(mcl)$Totals$Paid[2]
mcl_ultimate_incurred = summary(mcl)$Totals$Incurred[2]

actual_ultimate_paid = sum(payments_cumulative_square[, 20])

latest_paid = summary(mcl)$Totals$Paid[1]
latest_incurred = summary(mcl)$Totals$Incurred[1]

actual_OCL = actual_ultimate_paid - latest_paid
mcl_paid_OCL = mcl_ultimate_paid - latest_paid
mcl_incurred_OCL = mcl_ultimate_incurred - latest_paid
latest_OCL = latest_incurred - latest_paid

# Total Forecast Error (MCL)
abs(mcl_ultimate_paid - actual_ultimate_paid) / actual_ultimate_paid

# Total Forecast Error (Raw Latest Incurred)
abs(latest_incurred - actual_ultimate_paid) / actual_ultimate_paid

# OCL Forecast Error (MCL paid)
abs(mcl_paid_OCL - actual_OCL) / actual_OCL

# OCL Forecast Error (MCL incurred)
abs(mcl_incurred_OCL - actual_OCL) / actual_OCL

# OCL Forecast Error (Raw Latest Incurred)
abs(latest_OCL - actual_OCL) / actual_OCL

# Comparing paid chain ladder as well
cl_paid = chainladder(payments_cumulative_triangle)
cl_ultimate_paid = sum(predict(cl_paid)[, 20])
cl_paid_OCL = cl_ultimate_paid - latest_paid

# OCL Forecast Error (paid CL)
abs(cl_paid_OCL - actual_OCL) / actual_OCL


# and incurred chain ladder
cl_incurred = chainladder(incurred_cumulative_triangle)
cl_ultimate_incurred = sum(predict(cl_incurred)[, 20])
cl_incurred_OCL = cl_ultimate_incurred - latest_paid

abs(cl_incurred_OCL - actual_OCL) / actual_OCL
