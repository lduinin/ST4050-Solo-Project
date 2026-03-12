# ==============================================================================
# 02b_EDA_Post_Cleaning.R — Tables computed AFTER cleaning and splitting
# ==============================================================================
# REQUIRES: prepared_data.RData from 01_Data_Preparation.R
# Table 3 compares the learn/test split on cleaned data (capped claims/exposure).
# ==============================================================================

source("00_utils.R")
load("prepared_data.RData")

# ==============================================================================
# TABLE 3: Comparison of learning and test data sets (Page 11)
# ==============================================================================

calc_claim_dist <- function(data) {
  props <- sapply(0:4, function(x) mean(data$ClaimNb == x) * 100)
  freq  <- sum(data$ClaimNb) / sum(data$Exposure) * 100
  c(props, freq)
}

table3 <- data.frame(
  Claims   = c(0:4, "Frequency"),
  Learning = sprintf("%.3f%%", calc_claim_dist(learn)),
  Test     = sprintf("%.3f%%", calc_claim_dist(test))
)

print(table3, row.names = FALSE)

# ==============================================================================
# SAVE
# ==============================================================================

write.csv(table3, "table3_learn_test_comparison.csv", row.names = FALSE)
