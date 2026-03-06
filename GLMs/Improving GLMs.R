# ==============================================================================
# IMPROVED GLM MODELS - Beyond Model GLM1
# ==============================================================================

# Load required package for splines
library(splines)

# ==============================================================================
# MODEL GLM4: Add Key Interaction Terms
# ==============================================================================
time_glm4_start <- Sys.time()

glm4 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
              VehBrand + VehGas + DensityGLM + Region + AreaGLM +
              # Key interactions identified from exploratory analysis
              VehAgeGLM:VehBrand +           # New cars by brand (rental car effect)
              DrivAgeGLM:BonusMalusGLM +     # Age-experience interaction
              VehGas:VehPowerGLM,            # Fuel type and power interaction
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm4 <- as.numeric(difftime(Sys.time(), time_glm4_start, units = "secs"))
cat(sprintf("  Runtime: %.1fs, Parameters: %d\n", time_glm4, length(coef(glm4))))

#splines work


fit_spline_models <- function(data, spline_var, df_values) {
  # Initialize results data frame
  Spline_results <- data.frame(
    df = df_values,
    AIC = numeric(length(df_values)),
    BIC = numeric(length(df_values)),
    Deviance = numeric(length(df_values)),
    GCV = numeric(length(df_values))
  )
  
  # Loop through different degrees of freedom
  for (i in seq_along(df_values)) {
    df <- df_values[i]
    
    # Build formula based on df
    if (df == 1) {
      # Linear model (no spline)
      formula <- ClaimNb ~ get(spline_var)
    } else {
      # Natural spline with specified df
      formula <- ClaimNb ~ ns(get(spline_var), df = df)
    }
    
    # Fit the model (fixed: Poisson family, offset log(Exposure))
    model <- glm(formula = formula,
                 family = poisson(),
                 data = data,
                 offset = log(Exposure))
    
    # Store results
    Spline_results$AIC[i] <- AIC(model)
    Spline_results$BIC[i] <- BIC(model)
    Spline_results$Deviance[i] <- deviance(model)
    Spline_results$GCV[i] <- deviance(model) / (nobs(model) - df)
  }
  
  # Return results
  return(Spline_results)
}

Spline_results_Age <- fit_spline_models(
  data = learn,
  spline_var = "DrivAge",
  df_values = 1:5
)
Spline_results_BM <- fit_spline_models(
  data = learn,
  spline_var = "BonusMalusGLM",
  df_values = 1:3
)
Spline_results_Density <- fit_spline_models(
  data = learn,
  spline_var = "DensityGLM",
  df_values = 1:5
)
# View results
print(Spline_results_Age)
print(Spline_results_BM)
print(Spline_results_Density)



# ==============================================================================
# MODEL GLM5: Natural Cubic Splines for Continuous Variables
# ==============================================================================
time_glm5_start <- Sys.time()

glm5 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
              ns(DrivAge, df = 4) +              # Natural spline with 4 df for DrivAge
              ns(BonusMalusGLM, df = 4) +        # Natural spline with 4 df for BonusMalus
              VehBrand + VehGas + 
              DensityGLM +           # Natural spline with 3 df for log(Density)
              Region + AreaGLM,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm5 <- as.numeric(difftime(Sys.time(), time_glm5_start, units = "secs"))
cat(sprintf("  Runtime: %.1fs, Parameters: %d\n", time_glm5, length(coef(glm5))))

# ==============================================================================
# MODEL GLM6: Splines + Interactions (Kitchen Sink Model)
# ==============================================================================
time_glm6_start <- Sys.time()

glm6 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
              ns(DrivAge, df = 4) + 
              ns(BonusMalusGLM, df = 4) + 
              VehBrand + VehGas + 
              DensityGLM + 
              Region + AreaGLM +
              # Interactions
              VehAgeGLM:VehBrand +
              VehGas:VehPowerGLM,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm6 <- as.numeric(difftime(Sys.time(), time_glm6_start, units = "secs"))
cat(sprintf("  Runtime: %.1fs, Parameters: %d\n", time_glm6, length(coef(glm6))))

# ==============================================================================
# MODEL GLM7: Polynomial Terms (Alternative to Splines)
# ==============================================================================
time_glm7_start <- Sys.time()

glm7 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
              poly(DrivAge, 2) +                 # 2nd order polynomial for DrivAge
              poly(BonusMalusGLM, 2) +           # 2nd order polynomial for BonusMalus
              VehBrand + VehGas + 
              poly(DensityGLM, 2) +              # 2nd order polynomial for log(Density)
              Region + AreaGLM,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm7 <- as.numeric(difftime(Sys.time(), time_glm7_start, units = "secs"))
cat(sprintf("  Runtime: %.1fs, Parameters: %d\n", time_glm7, length(coef(glm7))))

# ==============================================================================
# MODEL GLM8: Best Model (Drop Area, Keep Splines + Key Interactions)
# ==============================================================================
time_glm8_start <- Sys.time()

glm8 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
              ns(DrivAge, df = 4) + 
              ns(BonusMalusGLM, df = 4) + 
              VehBrand + VehGas + 
              DensityGLM + 
              Region +
              # Key interactions only
              VehAgeGLM:VehBrand +
              VehGas:VehPowerGLM,
            family = poisson(),
            data = learn,
            offset = log(Exposure))

time_glm8 <- as.numeric(difftime(Sys.time(), time_glm8_start, units = "secs"))
cat(sprintf("  Runtime: %.1fs, Parameters: %d\n\n", time_glm8, length(coef(glm8))))

# ==============================================================================
# CALCULATE LOSSES FOR ALL IMPROVED MODELS
# ==============================================================================

# Add new models to the comparison
improved_models <- list(
  "Model GLM4 (interactions)" = glm4,
  "Model GLM5 (splines)" = glm5,
  "Model GLM6 (splines+interactions)" = glm6,
  "Model GLM8 (optimized)" = glm8
)

improved_runtimes <- c(time_glm4, time_glm5, time_glm6, time_glm8)

improved_results <- data.frame(
  Model = character(),
  Runtime = character(),
  Parameters = integer(),
  AIC = character(),
  InSample = character(),
  OutOfSample = character(),
  stringsAsFactors = FALSE
)

for (i in 1:length(improved_models)) {
  model_name <- names(improved_models)[i]
  model <- improved_models[[i]]
  
  # Get predictions
  learn_pred <- predict(model, type = "response") / learn$Exposure
  test_pred <- predict(model, newdata = test, type = "response") / test$Exposure
  
  # Calculate losses
  in_loss <- poisson_deviance(learn$ClaimNb, learn_pred, learn$Exposure)
  out_loss <- poisson_deviance(test$ClaimNb, test_pred, test$Exposure)
  
  # Get model info
  n_params <- length(coef(model))
  aic_val <- format(round(AIC(model)), big.mark = "'")
  runtime_str <- sprintf("%ds", round(improved_runtimes[i]))
  
  improved_results <- rbind(improved_results, data.frame(
    Model = model_name,
    Runtime = runtime_str,
    Parameters = n_params,
    AIC = aic_val,
    InSample = sprintf("%.5f", in_loss),
    OutOfSample = sprintf("%.5f", out_loss),
    stringsAsFactors = FALSE
  ))
}
print(results)
print(improved_results)
# ==============================================================================
# CREATE COMPARISON TABLE: Original vs Improved GLMs
# ==============================================================================

colnames(results) <- c("Model", "Runtime", "Parameters", "AIC", "InSample", "OutOfSample")
colnames(improved_results) <- c("Model", "Runtime", "Parameters", "AIC", "InSample", "OutOfSample")

# Combine original and improved results
comparison_results <- rbind(
  results[results$Model %in% c("homogeneous model", "Model GLM1", "Model GLM2", "Model GLM3"), ],
  improved_results
)

colnames(comparison_results) <- c("Model", "run time", "# param.", "AIC", 
                                  "in-sample loss", "out-of-sample loss")

print(comparison_results, row.names = FALSE)

# ==============================================================================
# IDENTIFY BEST MODEL
# ==============================================================================

# Convert losses back to numeric for comparison
comparison_results$out_numeric <- as.numeric(comparison_results$`out-of-sample loss`)


best_model_idx <- which.min(comparison_results$out_numeric)
best_model_name <- comparison_results$Model[best_model_idx]
best_loss <- comparison_results$out_numeric[best_model_idx]


cat("Best model (by out-of-sample loss):", best_model_name, "\n")
cat("Out-of-sample loss:", sprintf("%.5f", best_loss), "\n")
cat("Improvement over GLM1:", 
    sprintf("%.5f", as.numeric(results$OutOfSample[results$Model == "Model GLM1"]) - best_loss), 
    "units\n\n")

# ==============================================================================
# IDENTIFY BEST MODEL BY AIC
# ==============================================================================

# Remove formatting and convert AIC to numeric
comparison_results$AIC_numeric <- gsub("'", "", comparison_results$AIC)

# Convert to numeric
comparison_results$AIC_numeric <- as.numeric(comparison_results$AIC_numeric)

# Find best model (excluding NAs)
best_model_aic_idx <- which.min(comparison_results$AIC_numeric)
best_model_aic_name <- comparison_results$Model[best_model_aic_idx]
best_aic <- comparison_results$AIC_numeric[best_model_aic_idx]

cat("Best model (by AIC):", best_model_aic_name, "\n")
cat("AIC:", format(round(best_aic), big.mark = "'"), "\n\n")

# Show top 5 models by AIC
top5_aic <- comparison_results[order(comparison_results$AIC_numeric)[1:5], 
                               c("Model", "AIC", "# param.")]
cat("Top 5 models by AIC:\n")
print(top5_aic, row.names = FALSE)
cat("\n")

# Compare AIC improvement
glm1_aic <- as.numeric(gsub("'", "", results$AIC[results$Model == "Model GLM1"]))
aic_improvement <- glm1_aic - best_aic

cat("AIC improvement over GLM1:", format(round(aic_improvement), big.mark = "'"), "\n")
cat("(Lower AIC is better)\n\n")

# ==============================================================================
# SAVE IMPROVED MODELS
# ==============================================================================

save(glm4, glm5, glm6, glm8, 
     improved_results, comparison_results,
     file = "improved_glm_models.RData")

write.csv(comparison_results, 
          "comparison_original_vs_improved_glms.csv", 
          row.names = FALSE)


# ==============================================================================
# ANOVA COMPARISONS
# ==============================================================================

#GLM 1 vs GLM 4
print(anova(glm1, glm4, test = "Chisq"))

#GLM 1 vs GLM 5
print(anova(glm1, glm5, test = "Chisq"))

#GLM 5 vs GLM 6
print(anova(glm5, glm6, test = "Chisq"))

#GLM 6 vs GLM 8
print(anova(glm6, glm8, test = "Chisq"))

# ==============================================================================
# VISUALIZE SPLINE EFFECTS
# ==============================================================================


pdf("spline_effects.pdf", width = 12, height = 8)

par(mfrow = c(2, 3))

# Plot effects of splines from GLM5
# DrivAge effect
plot(learn$DrivAge, predict(glm5, type = "terms")[, "ns(DrivAge, df = 4)"],
     xlab = "Driver Age", ylab = "Partial Effect",
     main = "Natural Spline: Driver Age Effect (df=4)",
     pch = ".", col = rgb(0, 0, 0, 0.1))
lines(sort(unique(learn$DrivAge)), 
      predict(glm5, newdata = data.frame(
        DrivAge = sort(unique(learn$DrivAge)),
        BonusMalusGLM = median(learn$BonusMalusGLM),
        DensityGLM = median(learn$DensityGLM),
        VehPowerGLM = median(learn$VehPowerGLM),
        VehAgeGLM = levels(learn$VehAgeGLM)[1],
        VehBrand = levels(learn$VehBrand)[1],
        VehGas = levels(learn$VehGas)[1],
        Region = levels(learn$Region)[1],
        AreaGLM = median(learn$AreaGLM),
        Exposure = 1
      ), type = "terms")[, "ns(DrivAge, df = 4)"],
      col = "red", lwd = 2)

# BonusMalus effect
plot(learn$BonusMalusGLM, predict(glm5, type = "terms")[, "ns(BonusMalusGLM, df = 4)"],
     xlab = "Bonus-Malus Level", ylab = "Partial Effect",
     main = "Natural Spline: Bonus-Malus Effect (df=4)",
     pch = ".", col = rgb(0, 0, 0, 0.1))

# Density effect
plot(learn$DensityGLM, predict(glm5, type = "terms")[, "ns(DensityGLM, df = 3)"],
     xlab = "Log(Density)", ylab = "Partial Effect",
     main = "Natural Spline: Density Effect (df=3)",
     pch = ".", col = rgb(0, 0, 0, 0.1))

dev.off()
