# ==============================================================================
# 01_Data_Preparation.R — Load, Type Convert, Clean, Split
# Based on: Noll, Salzmann, Wüthrich (2020) Case Study
# ==============================================================================
# OUTPUT: 
#   - "raw_data.RData"      (after type conversions, BEFORE cleaning)
#   - "prepared_data.RData"  (after cleaning + train/test split)
# ==============================================================================

rm(list = ls())

RNGversion("3.5.0")
set.seed(100)

source("00_utils.R")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

freMTPL2freq <- read.csv("freMTPL2freq.csv")
str(freMTPL2freq)

# ==============================================================================
# 2. TYPE CONVERSIONS
# ==============================================================================

# Factors
freMTPL2freq$Area     <- as.factor(freMTPL2freq$Area)
freMTPL2freq$VehBrand <- as.factor(freMTPL2freq$VehBrand)
freMTPL2freq$VehGas   <- as.factor(freMTPL2freq$VehGas)
freMTPL2freq$Region   <- as.factor(freMTPL2freq$Region)

# Integers
freMTPL2freq$VehPower   <- as.integer(freMTPL2freq$VehPower)
freMTPL2freq$VehAge     <- as.integer(freMTPL2freq$VehAge)
freMTPL2freq$DrivAge    <- as.integer(freMTPL2freq$DrivAge)
freMTPL2freq$BonusMalus <- as.integer(freMTPL2freq$BonusMalus)
freMTPL2freq$Density    <- as.integer(freMTPL2freq$Density)

# Numerics
freMTPL2freq$ClaimNb  <- as.numeric(freMTPL2freq$ClaimNb)
freMTPL2freq$Exposure <- as.numeric(freMTPL2freq$Exposure)

str(freMTPL2freq)

# ==============================================================================
# 3. SAVE RAW (pre-cleaning) — for EDA tables that need uncapped data
# ==============================================================================

save(freMTPL2freq, file = "raw_data.RData")

# ==============================================================================
# 4. DATA CLEANING (Pages 3-4 of paper)
# ==============================================================================

# Cap exposures at 1 year (exposures > 1 are data errors)
freMTPL2freq$Exposure <- pmin(freMTPL2freq$Exposure, 1)

# Cap claim counts at 4 (ClaimNb > 4 are likely data errors)
freMTPL2freq$ClaimNb <- pmin(freMTPL2freq$ClaimNb, 4)

# ==============================================================================
# 5. TRAIN/TEST SPLIT — 90:10 (Listing 2, Page 11)
# ==============================================================================

n <- nrow(freMTPL2freq)
learn_indices <- sample(1:n, round(0.9 * n), replace = FALSE)

learn <- freMTPL2freq[learn_indices, ]
test  <- freMTPL2freq[-learn_indices, ]

nrow(learn)
nrow(test)

# ==============================================================================
# 6. SAVE CLEANED + SPLIT
# ==============================================================================

save(freMTPL2freq, learn, test, file = "prepared_data.RData")
