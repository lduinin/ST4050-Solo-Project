# ==============================================================================
# French Motor Third-Party Liability Claims - Frequency Modeling Tables
# Complete Recreation Script
# Based on: Noll, Salzmann, Wüthrich (2020) Case Study
# ==============================================================================

# Clear environment
rm(list = ls())

# Set random seed for reproducibility (as per paper)
RNGversion("3.5.0")
set.seed(100)

# ==============================================================================
# 1. LOAD REQUIRED PACKAGES
# ==============================================================================
getwd()
load("freMTPL2freq.csv")
# Install CASdatasets if needed (contains the freMTPL2freq dataset)
if (!require("CASdatasets")) {
  install.packages("CASdatasets", 
                   repos = "http://dutangc.free.fr/pub/RRepos/", 
                   type = "source")
}

# Load required libraries
library(CASdatasets)  # For the dataset
library(stats)        # For GLM
library(rpart)        # For regression trees
library(rpart.plot)   # For plotting trees

# Check version (paper uses CASdatasets 1.0-8)
cat("CASdatasets version:", as.character(packageVersion("CASdatasets")), "\n\n")
# ==============================================================================
# 3. CREATE TABLE 2: Split of Portfolio w.r.t. Number of Claims (Page 4)
# ==============================================================================

#TABLE 1: Split of Portfolio w.r.t. Number of Claims

# Calculate statistics for each claim count
claim_counts <- sort(unique(freMTPL2freq$ClaimNb))

# Create table
table1 <- data.frame(
  Claims = claim_counts,
  Policies = sapply(claim_counts, function(x) sum(freMTPL2freq$ClaimNb == x)),
  Exposure = sapply(claim_counts, function(x) {
    sum(freMTPL2freq$Exposure[freMTPL2freq$ClaimNb == x])
  })
)
print(table1)

# Calculate overall portfolio statistics
total_policies <- nrow(freMTPL2freq)
total_exposure <- sum(freMTPL2freq$Exposure)
total_claims <- sum(freMTPL2freq$ClaimNb)
portfolio_frequency <- total_claims / total_exposure

cat("Summary Statistics:\n")
cat("-------------------\n")
cat("Total policies:     ", format(total_policies, big.mark = "'"), "\n")
cat("Total exposure:     ", format(round(total_exposure, 1), big.mark = "'"), "\n")
cat("Total claims:       ", format(total_claims, big.mark = "'"), "\n")
cat("Portfolio frequency:", sprintf("%.6f (%.4f%%)", portfolio_frequency, 
                                    portfolio_frequency * 100), "\n\n")

# ==============================================================================
# 3. LOAD AND CLEAN DATA
# ==============================================================================

data(freMTPL2freq)

cat("Original dataset dimensions:", nrow(freMTPL2freq), "rows,", 
    ncol(freMTPL2freq), "columns\n\n")

# Display structure (as in Listing 1 of the paper)
str(freMTPL2freq)
# Convert variables to appropriate types

#convert to factors
freMTPL2freq$Area <- as.factor(freMTPL2freq$Area)
freMTPL2freq$VehBrand <- as.factor(freMTPL2freq$VehBrand)
freMTPL2freq$VehGas <- as.factor(freMTPL2freq$VehGas)
freMTPL2freq$Region <- as.factor(freMTPL2freq$Region)

#ensure integer
freMTPL2freq$VehPower <- as.integer(freMTPL2freq$VehPower)
freMTPL2freq$VehAge <- as.integer(freMTPL2freq$VehAge)
freMTPL2freq$DrivAge <- as.integer(freMTPL2freq$DrivAge)
freMTPL2freq$BonusMalus <- as.integer(freMTPL2freq$BonusMalus)


# Density: ensure it's numeric (integer)
freMTPL2freq$Density <- as.integer(freMTPL2freq$Density)

# ClaimNb: ensure it's numeric
freMTPL2freq$ClaimNb <- as.numeric(freMTPL2freq$ClaimNb)

# Exposure: ensure it's numeric
freMTPL2freq$Exposure <- as.numeric(freMTPL2freq$Exposure)

# Verify the conversions
str(freMTPL2freq)
# DATA CLEANING (as described on pages 3-4 of the paper)


# ==============================================================================
# CREATE TABLE 2: Correlations in Feature Components (Page 9)
# ==============================================================================

# Select the continuous/ordinal variables for correlation analysis
# Note: We need to convert categorical variables to numeric for correlation
Data_corr <- freMTPL2freq

# Convert Area from factor to integer (A=1, B=2, etc.)
Data_corr$Area_num <- as.integer(Data_corr$Area)

# VehPower is already integer
Data_corr$VehPower_num <- as.integer(Data_corr$VehPower)

# VehAge is already integer
Data_corr$VehAge_num <- as.integer(Data_corr$VehAge)

# DrivAge is already integer
Data_corr$DrivAge_num <- as.integer(Data_corr$DrivAge)

# BonusMalus is already integer
Data_corr$BonusMalus_num <- as.integer(Data_corr$BonusMalus)

# Density is already integer
Data_corr$Density_num <- as.integer(Data_corr$Density)

# Create data frame with just the numeric variables for correlation
corr_data <- data.frame(
  Area = Data_corr$Area_num,
  VehPower = Data_corr$VehPower_num,
  VehAge = Data_corr$VehAge_num,
  DrivAge = Data_corr$DrivAge_num,
  BonusMalus = Data_corr$BonusMalus_num,
  Density = Data_corr$Density_num
)

# Calculate Pearson correlation (for top-right of table)
pearson_corr <- cor(corr_data, method = "pearson")

# Calculate Spearman correlation (for bottom-left of table)
spearman_corr <- cor(corr_data, method = "spearman")

# Create the combined correlation matrix as in the paper
# Top-right: Pearson, Bottom-left: Spearman
table2 <- pearson_corr
for (i in 1:nrow(table2)) {
  for (j in 1:ncol(table2)) {
    if (i > j) {
      # Bottom-left triangle: Spearman's rho
      table2[i, j] <- spearman_corr[i, j]
    }
    if (i == j) {
      # Diagonal: leave empty or set to NA for display
      table2[i, j] <- NA
    }
  }
}

# Round to 2 decimal places for display
table2_display <- round(table2, 2)

# Format for display (replace NA on diagonal with blank)
table2_formatted <- table2_display
table2_formatted[is.na(table2_formatted)] <- ""

# Print the table
print(table2_formatted)

# Save Table 2 to CSV
write.csv(table2_display, 
          "table2_correlations.csv", 
          row.names = TRUE)
# ========================
# DATA CLEANING 2
# =============================

# 1. Cap exposures at 1 year (exposures > 1 are data errors)
freMTPL2freq$Exposure <- pmin(freMTPL2freq$Exposure, 1)

# 2. Cap claim counts at 4 (ClaimNb > 4 are likely data errors)
freMTPL2freq$ClaimNb <- pmin(freMTPL2freq$ClaimNb, 4)

# ==============================================================================
# 4. CREATE LEARNING AND TEST DATASETS (Listing 2, Page 11)
# ==============================================================================
# 90:10 LEARNING TEST SPLIT 

# Create 90/10 split for learning/test
n <- nrow(freMTPL2freq)
learn_indices <- sample(1:n, round(0.9 * n), replace = FALSE)

learn <- freMTPL2freq[learn_indices, ]
test <- freMTPL2freq[-learn_indices, ]

cat("Learning data:  ", nrow(learn), "policies\n")
cat("Test data:      ", nrow(test), "policies\n\n")

# ==============================================================================
# 5. CREATE TABLE 3: Comparison of Learning and Test Data Sets (Page 11)
# ==============================================================================

# Function to calculate claim distribution
calc_claim_dist <- function(data) {
  claims <- 0:4
  props <- sapply(claims, function(x) {
    mean(data$ClaimNb == x) * 100
  })
  freq <- sum(data$ClaimNb) / sum(data$Exposure) * 100
  c(props, freq)
}

# Calculate for learning and test sets
learn_stats <- calc_claim_dist(learn)
test_stats <- calc_claim_dist(test)

# Create table
table3 <- data.frame(
  Claims = c(0:4, "Frequency"),
  Learning = sprintf("%.3f%%", learn_stats),
  Test = sprintf("%.3f%%", test_stats)
)

print(table3, row.names = FALSE)
