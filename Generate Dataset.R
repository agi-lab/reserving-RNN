# Code slightly adapted from SPLICE paper (section 5.1.1)

library(SPLICE)
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
test_inflated <- claim_history(test_claims, major, minor,
                               base_inflation_vector = rep(1.02^(1/4) - 1, 
                               times = 80))

# transactional data
test_incurred_dataset_noInf <- generate_incurred_dataset(test_claims, test)
test_incurred_dataset_inflated <- generate_incurred_dataset(test_claims, 
                                                            test_inflated)

write.csv(test_incurred_dataset_noInf, 
'Datasets/R Outputs/test_incurred_dataset_noInf.csv')

write.csv(test_incurred_dataset_inflated,
'Datasets/R Outputs/test_incurred_dataset_inflated.csv')