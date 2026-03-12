# ==============================================================================
# 04_Improve_GLMs.R — Improved GLMs (Interactions, Splines, Polynomials)
# ==============================================================================
# REQUIRES: benchmark_glms.RData from 03_Replicate_GLMs.R
# OUTPUT: saves "improved_glms.RData" with all improved models added
# ==============================================================================

source("00_utils.R")
load("benchmark_glms.RData")

library(splines)

# ==============================================================================
# SPLINE DEGREE-OF-FREEDOM SELECTION
# ==============================================================================
# Fit univariate spline models at different df to find the best df
# for each continuous variable before putting them into the full model.

fit_spline_models <- function(data, spline_var, df_values) {
  results <- data.frame(df = df_values, AIC = NA, BIC = NA, Deviance = NA)
  
  for (i in seq_along(df_values)) {
    d <- df_values[i]
    
    if (d == 1) {
      formula <- as.formula(paste("ClaimNb ~", spline_var))
    } else {
      formula <- as.formula(paste0("ClaimNb ~ ns(", spline_var, ", df = ", d, ")"))
    }
    
    model <- glm(formula, family = poisson(), data = data, offset = log(Exposure))
    results$AIC[i]      <- AIC(model)
    results$BIC[i]      <- BIC(model)
    results$Deviance[i] <- deviance(model)
  }
  
  return(results)
}

# DrivAge spline selection
spline_drivage <- fit_spline_models(learn, "DrivAge", 1:5)
print(spline_drivage)

# BonusMalus spline selection
spline_bm <- fit_spline_models(learn, "BonusMalusGLM", 1:3)
print(spline_bm)

# Density spline selection
spline_density <- fit_spline_models(learn, "DensityGLM", 1:5)
print(spline_density)

# ==============================================================================
# MODEL GLM4: Interactions only
# ==============================================================================

time_glm4_start <- Sys.time()
glm4 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
               VehBrand + VehGas + DensityGLM + Region + AreaGLM +
               VehAgeGLM:VehBrand +
               DrivAgeGLM:BonusMalusGLM +
               VehGas:VehPowerGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm4 <- as.numeric(difftime(Sys.time(), time_glm4_start, units = "secs"))

# ==============================================================================
# MODEL GLM5: Natural splines for continuous variables
# ==============================================================================
# Uses DrivAge df=4, BonusMalus df=4 (based on spline selection above)
# DensityGLM kept linear (spline selection showed minimal improvement)

time_glm5_start <- Sys.time()
glm5 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
               ns(DrivAge, df = 4) + 
               ns(BonusMalusGLM, df = 4) + 
               VehBrand + VehGas + DensityGLM + Region + AreaGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm5 <- as.numeric(difftime(Sys.time(), time_glm5_start, units = "secs"))

# ==============================================================================
# MODEL GLM6: Splines + Interactions
# ==============================================================================

time_glm6_start <- Sys.time()
glm6 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
               ns(DrivAge, df = 4) + 
               ns(BonusMalusGLM, df = 4) + 
               VehBrand + VehGas + DensityGLM + Region + AreaGLM +
               VehAgeGLM:VehBrand +
               VehGas:VehPowerGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm6 <- as.numeric(difftime(Sys.time(), time_glm6_start, units = "secs"))

# ==============================================================================
# MODEL GLM7: Polynomials (alternative to splines)
# ==============================================================================

time_glm7_start <- Sys.time()
glm7 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
               poly(DrivAge, 2) + 
               poly(BonusMalusGLM, 2) + 
               VehBrand + VehGas + 
               poly(DensityGLM, 2) + 
               Region + AreaGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm7 <- as.numeric(difftime(Sys.time(), time_glm7_start, units = "secs"))

# ==============================================================================
# MODEL GLM8: Drop Area, keep splines + key interactions
# ==============================================================================

time_glm8_start <- Sys.time()
glm8 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + 
               ns(DrivAge, df = 4) + 
               ns(BonusMalusGLM, df = 4) + 
               VehBrand + VehGas + DensityGLM + Region +
               VehAgeGLM:VehBrand +
               VehGas:VehPowerGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm8 <- as.numeric(difftime(Sys.time(), time_glm8_start, units = "secs"))

# ==============================================================================
# RESULTS — ALL IMPROVED MODELS (including GLM7 this time)
# ==============================================================================

improved_models  <- list(glm4, glm5, glm6, glm7, glm8)
improved_names   <- c("GLM4 (interactions)", "GLM5 (splines)", 
                       "GLM6 (splines+interact)", "GLM7 (polynomials)",
                       "GLM8 (optimised)")
improved_times   <- c(time_glm4, time_glm5, time_glm6, time_glm7, time_glm8)

improved_results <- do.call(rbind, lapply(seq_along(improved_models), function(i) {
  evaluate_model(improved_models[[i]], improved_names[i], learn, test, improved_times[i])
}))

# Combined comparison: benchmark + improved
comparison <- rbind(results, improved_results)
comparison <- comparison[order(comparison$OutOfSample), ]

print(comparison, row.names = FALSE)

# ==============================================================================
# ANOVA COMPARISONS
# ==============================================================================

# GLM1 vs GLM4 (do interactions help?)
print(anova(glm1, glm4, test = "Chisq"))

# GLM1 vs GLM5 (do splines help?)
print(anova(glm1, glm5, test = "Chisq"))

# GLM5 vs GLM6 (do interactions help on top of splines?)
print(anova(glm5, glm6, test = "Chisq"))

# GLM6 vs GLM8 (is dropping Area justified?)
print(anova(glm8, glm6, test = "Chisq"))

# ==============================================================================
# SPLINE EFFECT PLOTS
# ==============================================================================

pdf("spline_effects.pdf", width = 12, height = 5)
par(mfrow = c(1, 3))

# DrivAge partial effect
plot(learn$DrivAge, predict(glm5, type = "terms")[, "ns(DrivAge, df = 4)"],
     xlab = "Driver Age", ylab = "Partial Effect",
     main = "Natural Spline: Driver Age (df=4)",
     pch = ".", col = rgb(0, 0, 0, 0.1))

# BonusMalus partial effect
plot(learn$BonusMalusGLM, predict(glm5, type = "terms")[, "ns(BonusMalusGLM, df = 4)"],
     xlab = "Bonus-Malus Level", ylab = "Partial Effect",
     main = "Natural Spline: Bonus-Malus (df=4)",
     pch = ".", col = rgb(0, 0, 0, 0.1))

# DensityGLM partial effect (linear in GLM5)
plot(learn$DensityGLM, predict(glm5, type = "terms")[, "DensityGLM"],
     xlab = "log(Density)", ylab = "Partial Effect",
     main = "Linear: log(Density)",
     pch = ".", col = rgb(0, 0, 0, 0.1))

dev.off()

# ==============================================================================
# SAVE — everything needed for regularisation and further analysis
# ==============================================================================

save(learn, test, 
     glm_homog, glm1, glm2, glm3,
     glm4, glm5, glm6, glm7, glm8,
     results, improved_results, comparison,
     file = "improved_glms.RData")

write.csv(comparison, "comparison_all_glms.csv", row.names = FALSE)
