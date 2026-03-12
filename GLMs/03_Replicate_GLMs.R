# ==============================================================================
# 03_Replicate_GLMs.R — Benchmark GLMs (Section 3.2, Pages 14-18)
# ==============================================================================
# REQUIRES: prepared_data.RData from 01_Data_Preparation.R
# OUTPUT: saves "benchmark_glms.RData" with learn, test (with GLM features),
#         glm_homog, glm1, glm2, glm3, and results table
# ==============================================================================

source("00_utils.R")
load("prepared_data.RData")

# ==============================================================================
# 1. FEATURE PRE-PROCESSING (Listing 3, Page 14)
# ==============================================================================

# Area: continuous (as integer 1-6)
learn$AreaGLM <- as.integer(learn$Area)
test$AreaGLM  <- as.integer(test$Area)

# VehPower: categorical (merge >= 9)
learn$VehPowerGLM <- as.factor(pmin(learn$VehPower, 9))
test$VehPowerGLM  <- as.factor(pmin(test$VehPower, 9))

# VehAge: 3 categorical classes [0,1), [1,10], (10,inf)
VehAgeGLM_map <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
learn$VehAgeGLM <- factor(VehAgeGLM_map[learn$VehAge + 1, 2], levels = c("2", "1", "3"))
test$VehAgeGLM  <- factor(VehAgeGLM_map[test$VehAge + 1, 2],  levels = c("2", "1", "3"))

# DrivAge: 7 categorical classes
DrivAgeGLM_map <- cbind(c(18:100), c(
  rep(1, 21-18), rep(2, 26-21), rep(3, 31-26),
  rep(4, 41-31), rep(5, 51-41), rep(6, 71-51),
  rep(7, 100-71+1)
))
learn$DrivAgeGLM <- factor(DrivAgeGLM_map[learn$DrivAge - 17, 2], levels = c("5","1","2","3","4","6","7"))
test$DrivAgeGLM  <- factor(DrivAgeGLM_map[test$DrivAge - 17, 2],  levels = c("5","1","2","3","4","6","7"))

# BonusMalus: continuous (capped at 150)
learn$BonusMalusGLM <- as.integer(pmin(learn$BonusMalus, 150))
test$BonusMalusGLM  <- as.integer(pmin(test$BonusMalus, 150))

# Density: log-density as continuous
learn$DensityGLM <- as.numeric(log(learn$Density))
test$DensityGLM  <- as.numeric(log(test$Density))

# Region: set reference to R24
learn$Region <- factor(learn$Region, levels = c("R24", setdiff(levels(learn$Region), "R24")))
test$Region  <- factor(test$Region,  levels = c("R24", setdiff(levels(test$Region), "R24")))

# ==============================================================================
# 2. FIT GLM MODELS
# ==============================================================================

# Homogeneous model (intercept only)
time_homog_start <- Sys.time()
glm_homog <- glm(ClaimNb ~ 1, family = poisson(), data = learn, offset = log(Exposure))
time_homog <- as.numeric(difftime(Sys.time(), time_homog_start, units = "secs"))

# Model GLM1: all features
time_glm1_start <- Sys.time()
glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
               VehBrand + VehGas + DensityGLM + Region + AreaGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm1 <- as.numeric(difftime(Sys.time(), time_glm1_start, units = "secs"))

# Model GLM2: drop Area
time_glm2_start <- Sys.time()
glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
               VehBrand + VehGas + DensityGLM + Region,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm2 <- as.numeric(difftime(Sys.time(), time_glm2_start, units = "secs"))

# Model GLM3: drop Area and VehBrand
time_glm3_start <- Sys.time()
glm3 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
               VehGas + DensityGLM + Region,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm3 <- as.numeric(difftime(Sys.time(), time_glm3_start, units = "secs"))

# ==============================================================================
# 3. RESULTS TABLE (Table 5, Page 18)
# ==============================================================================

models_list <- list(
  "Homogeneous model" = glm_homog,
  "Model GLM1"        = glm1,
  "Model GLM2"        = glm2,
  "Model GLM3"        = glm3
)
runtimes <- c(time_homog, time_glm1, time_glm2, time_glm3)

results <- do.call(rbind, lapply(seq_along(models_list), function(i) {
  evaluate_model(models_list[[i]], names(models_list)[i], learn, test, runtimes[i])
}))

print(results, row.names = FALSE)

# ==============================================================================
# 4. SAVE — everything downstream scripts need
# ==============================================================================

save(learn, test, glm_homog, glm1, glm2, glm3, results,
     file = "benchmark_glms.RData")

write.csv(results, "table5_glm_results.csv", row.names = FALSE)
