# ==============================================================================
# 6. FEATURE PRE-PROCESSING FOR GLM (Listing 3, Page 14)
# ==============================================================================

# Area: continuous (as integer 1-6)
learn$AreaGLM <- as.integer(learn$Area)
test$AreaGLM <- as.integer(test$Area)

# VehPower: categorical (merge >=9)
learn$VehPowerGLM <- as.factor(pmin(learn$VehPower, 9))
test$VehPowerGLM <- as.factor(pmin(test$VehPower, 9))

# VehAge: 3 categorical classes [0,1), [1,10], (10,∞)
learn$VehAgeGLM <- factor(VehAgeGLM[learn$VehAge + 1, 2], levels = c("2", "1", "3"))
test$VehAgeGLM  <- factor(VehAgeGLM[test$VehAge + 1, 2],  levels = c("2", "1", "3"))

# DrivAge: 7 categorical classes
DrivAgeGLM <- cbind(c(18:100), c(rep(1, 21-18), rep(2, 26-21), rep(3, 31-26), 
                                 rep(4, 41-31), rep(5, 51-41), rep(6, 71-51), 
                                 rep(7, 100-71+1)))
learn$DrivAgeGLM <- factor(DrivAgeGLM[learn$DrivAge - 17, 2], levels = c("5","1","2","3","4","6","7"))
test$DrivAgeGLM  <- factor(DrivAgeGLM[test$DrivAge - 17, 2],  levels = c("5","1","2","3","4","6","7"))
# BonusMalus: continuous (capped at 150)
learn$BonusMalusGLM <- as.integer(pmin(learn$BonusMalus, 150))
test$BonusMalusGLM <- as.integer(pmin(test$BonusMalus, 150))

# Density: log-density as continuous
learn$DensityGLM <- as.numeric(log(learn$Density))
test$DensityGLM <- as.numeric(log(test$Density))

# Region: categorical (set reference to R24)
learn$Region <- factor(learn$Region, levels = c("R24", setdiff(levels(learn$Region), "R24")))
test$Region  <- factor(test$Region,  levels = c("R24", setdiff(levels(test$Region), "R24")))
# ==============================================================================
# 7. FIT GLM MODELS (Section 3.2, Pages 15-18)
# ==============================================================================

#homog GLM
time_homog_start <- Sys.time()

glm_homog <- glm(ClaimNb ~ 1,  # Intercept only
                 family = poisson(),
                 data = learn,
                 offset = log(Exposure))

time_homog <- as.numeric(difftime(Sys.time(), time_homog_start, units = "secs"))

# Model GLM1: All features
time_glm1_start <- Sys.time()

glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
              VehBrand + VehGas + DensityGLM + Region + AreaGLM,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm1 <- as.numeric(difftime(Sys.time(), time_glm1_start, units = "secs"))

# Model GLM2: Drop Area
cat("Fitting Model GLM2 (drop Area)...\n")
time_glm2_start <- Sys.time()

glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
              VehBrand + VehGas + DensityGLM + Region,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm2 <- as.numeric(difftime(Sys.time(), time_glm2_start, units = "secs"))

# Model GLM3: Drop Area and VehBrand
cat("Fitting Model GLM3 (drop Area and VehBrand)...\n")
time_glm3_start <- Sys.time()

glm3 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
              VehGas + DensityGLM + Region,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm3 <- as.numeric(difftime(Sys.time(), time_glm3_start, units = "secs"))


# ==============================================================================
# 8. CALCULATE DEVIANCE LOSSES (Section 2.2, Pages 10-12)
# ==============================================================================

# Function to calculate Poisson deviance loss
poisson_deviance <- function(y, mu, w) {
  # y = observed counts, mu = predicted means, w = exposures
  # Returns average deviance loss (in units of 10^-2)
  
  idx_zero <- (y == 0)
  dev <- numeric(length(y))
  
  dev[idx_zero] <- 2 * mu[idx_zero] * w[idx_zero]
  dev[!idx_zero] <- 2 * y[!idx_zero] * 
    (log(y[!idx_zero]/(mu[!idx_zero] * w[!idx_zero]))) + 
    2 * (mu[!idx_zero] * w[!idx_zero] - y[!idx_zero])
  
  mean(dev) * 100  # Return in units of 10^-2
}

# Homogeneous model (intercept only)
lambda_homog <- sum(learn$ClaimNb) / sum(learn$Exposure)

# Calculate losses for all models
models_list <- list(
  "homogeneous model" = glm_homog,
  "Model GLM1" = glm1,
  "Model GLM2" = glm2,
  "Model GLM3" = glm3
)
runtimes <- c(time_homog, time_glm1, time_glm2, time_glm3)
results <- data.frame(
  Model = character(),
  Runtime = numeric(),
  Parameters = integer(),
  AIC = numeric(),
  InSample = numeric(),
  OutOfSample = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:length(models_list)) {
  model_name <- names(models_list)[i]
  model <- models_list[[i]]
  
  # Get predictions
  learn_pred <- predict(model, type = "response") / learn$Exposure
  test_pred <- predict(model, newdata = test, type = "response") / test$Exposure
  
  # Calculate losses
  in_loss <- poisson_deviance(learn$ClaimNb, learn_pred, learn$Exposure)
  out_loss <- poisson_deviance(test$ClaimNb, test_pred, test$Exposure)
  
  # Get model info
  n_params <- length(coef(model))
  aic_val <- format(round(AIC(model)), big.mark = "'")
  runtime_str <- sprintf("%ds", round(runtimes[i]))
  
  results <- rbind(results, data.frame(
    Model = model_name,
    Runtime = runtime_str,
    Parameters = n_params,
    AIC = aic_val,
    InSample = sprintf("%.5f", in_loss),
    OutOfSample = sprintf("%.5f", out_loss),
    stringsAsFactors = FALSE
  ))
}

# ==============================================================================
# 9. CREATE TABLE 5: GLM Results Comparison (Page 18)
# ==============================================================================
colnames(results) <- c("Model", "run time", "# param.", "AIC", 
                       "in-sample loss", "out-of-sample loss")

print(results, row.names = FALSE)

# ==============================================================================
# 10. SAVE RESULTS AND MODELS
# ==============================================================================

# Save to outputs directory
save(freMTPL2freq, learn, test, glm1, glm2, glm3,
     table1, table3, results,
     file = "frequency_models.RData")

# Save tables as CSV
write.csv(table1, "table1_portfolio_split.csv", 
          row.names = FALSE)
write.csv(table3, "table3_learn_test_comparison.csv", 
          row.names = FALSE)
write.csv(results, "table5_glm_results.csv", 
          row.names = FALSE)
