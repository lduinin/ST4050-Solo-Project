# ==============================================================================
# 02a_EDA_Pre_Cleaning.R — Tables computed on RAW data (before capping)
# ==============================================================================
# REQUIRES: raw_data.RData from 01_Data_Preparation.R
# These tables match the paper which uses uncapped claim counts and exposures.
# ==============================================================================

source("00_utils.R")
load("raw_data.RData")

# ==============================================================================
# TABLE 1: Split of portfolio w.r.t. number of claims (Page 4)
# ==============================================================================
# Uses raw ClaimNb (including values > 4) to match the paper

claim_counts <- sort(unique(freMTPL2freq$ClaimNb))

table1 <- data.frame(
  Claims   = claim_counts,
  Policies = sapply(claim_counts, function(x) sum(freMTPL2freq$ClaimNb == x)),
  Exposure = sapply(claim_counts, function(x) sum(freMTPL2freq$Exposure[freMTPL2freq$ClaimNb == x]))
)

print(table1)

# Portfolio summary
total_policies      <- nrow(freMTPL2freq)
total_exposure      <- sum(freMTPL2freq$Exposure)
total_claims        <- sum(freMTPL2freq$ClaimNb)
portfolio_frequency <- total_claims / total_exposure

# ==============================================================================
# TABLE 2: Correlations in feature components (Page 9)
# ==============================================================================
# Pearson (upper triangle), Spearman (lower triangle)
# Computed on raw data before any capping

corr_data <- data.frame(
  Area       = as.integer(freMTPL2freq$Area),
  VehPower   = freMTPL2freq$VehPower,
  VehAge     = freMTPL2freq$VehAge,
  DrivAge    = freMTPL2freq$DrivAge,
  BonusMalus = freMTPL2freq$BonusMalus,
  Density    = freMTPL2freq$Density
)

pearson_corr  <- cor(corr_data, method = "pearson")
spearman_corr <- cor(corr_data, method = "spearman")

# Combine: upper = Pearson, lower = Spearman, diagonal = NA
table2 <- pearson_corr
for (i in 1:nrow(table2)) {
  for (j in 1:ncol(table2)) {
    if (i > j) table2[i, j] <- spearman_corr[i, j]
    if (i == j) table2[i, j] <- NA
  }
}

print(round(table2, 2))

# ==============================================================================
# SAVE
# ==============================================================================

write.csv(table1, "table1_portfolio_split.csv", row.names = FALSE)
write.csv(round(table2, 2), "table2_correlations.csv", row.names = TRUE)
