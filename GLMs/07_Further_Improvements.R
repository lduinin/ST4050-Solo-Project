# ==============================================================================
# 07_Further_Improvements.R — GAMs, Region Grouping, Higher-Order Interactions,
#                              and Additional Evaluation Metrics
# ==============================================================================
# REQUIRES: improved_glms.RData from 04_Improve_GLMs.R
#           regularisation_results.RData from 06_Regularisation.R
#           regression_trees.RData from 05_Regression_Trees.R
# OUTPUT: saves "further_improvements.RData"
# ==============================================================================

source("00_utils.R")
load("improved_glms.RData")
load("regularisation_results.RData")
load("regression_trees.RData")

if (!require("mgcv")) install.packages("mgcv")
library(mgcv)
library(glmnet)
library(splines)

# ==============================================================================
# HELPER FUNCTIONS: GINI, CALIBRATION
# ==============================================================================

gini_coefficient <- function(actual, predicted, exposure) {
  ord <- order(predicted)
  cum_actual   <- cumsum(actual[ord]) / sum(actual)
  cum_exposure <- cumsum(exposure[ord]) / sum(exposure)
  n <- length(cum_actual)
  auc <- sum((cum_exposure[2:n] - cum_exposure[1:(n-1)]) *
               (cum_actual[2:n] + cum_actual[1:(n-1)])) / 2
  
  ord_perfect <- order(actual / exposure)
  cum_actual_p   <- cumsum(actual[ord_perfect]) / sum(actual)
  cum_exposure_p <- cumsum(exposure[ord_perfect]) / sum(exposure)
  auc_perfect <- sum((cum_exposure_p[2:n] - cum_exposure_p[1:(n-1)]) *
                       (cum_actual_p[2:n] + cum_actual_p[1:(n-1)])) / 2
  
  (auc - 0.5) / (auc_perfect - 0.5)
}

calibration_ratio <- function(actual, predicted, exposure) {
  sum(predicted * exposure) / sum(actual)
}

# ==============================================================================
# PART 1: EXTENDED METRICS FOR ALL MODELS (GLMs + Regularised)
# ==============================================================================

all_glm_models <- list(
  "Homogeneous"             = glm_homog,
  "GLM1 (benchmark)"        = glm1,
  "GLM4 (interactions)"     = glm4,
  "GLM5 (splines)"          = glm5,
  "GLM6 (splines+interact)" = glm6,
  "GLM7 (polynomials)"      = glm7,
  "GLM8 (optimised)"        = glm8,
  "GLM9 (3-way)"            = glm9
)

extended_metrics <- do.call(rbind, lapply(names(all_glm_models), function(nm) {
  mod <- all_glm_models[[nm]]
  pred_learn <- predict(mod, type = "response") / learn$Exposure
  pred_test  <- predict(mod, newdata = test, type = "response") / test$Exposure
  data.frame(
    Model = nm, Type = "GLM",
    OutOfSample = round(poisson_deviance(test$ClaimNb, pred_test, test$Exposure), 5),
    Gini_train  = round(gini_coefficient(learn$ClaimNb, pred_learn, learn$Exposure), 4),
    Gini_test   = round(gini_coefficient(test$ClaimNb, pred_test, test$Exposure), 4),
    Calib_train = round(calibration_ratio(learn$ClaimNb, pred_learn, learn$Exposure), 4),
    Calib_test  = round(calibration_ratio(test$ClaimNb, pred_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

# --- Regularised models ---

reg_model_list <- list(
  "Lasso (lambda.min)"       = list(cv = cv_lasso, s = "lambda.min"),
  "Lasso (lambda.1se)"       = list(cv = cv_lasso, s = "lambda.1se"),
  "Ridge (lambda.min)"       = list(cv = cv_ridge, s = "lambda.min"),
  "Ridge (lambda.1se)"       = list(cv = cv_ridge, s = "lambda.1se"),
  "Elastic Net (lambda.min)" = list(cv = cv_enet,  s = "lambda.min"),
  "Elastic Net (lambda.1se)" = list(cv = cv_enet,  s = "lambda.1se")
)

reg_metrics <- do.call(rbind, lapply(names(reg_model_list), function(nm) {
  cv_mod <- reg_model_list[[nm]]$cv
  s_val  <- reg_model_list[[nm]]$s
  
  pred_learn <- as.numeric(predict(cv_mod, newx = X_learn, s = s_val,
                                   newoffset = offset_learn, type = "response"))
  pred_test  <- as.numeric(predict(cv_mod, newx = X_test, s = s_val,
                                   newoffset = offset_test, type = "response"))
  
  mu_learn <- pred_learn / learn$Exposure
  mu_test  <- pred_test  / test$Exposure
  
  data.frame(
    Model = nm, Type = "Regularised",
    OutOfSample = round(poisson_deviance(test$ClaimNb, mu_test, test$Exposure), 5),
    Gini_train  = round(gini_coefficient(learn$ClaimNb, mu_learn, learn$Exposure), 4),
    Gini_test   = round(gini_coefficient(test$ClaimNb, mu_test, test$Exposure), 4),
    Calib_train = round(calibration_ratio(learn$ClaimNb, mu_learn, learn$Exposure), 4),
    Calib_test  = round(calibration_ratio(test$ClaimNb, mu_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

extended_metrics <- rbind(extended_metrics, reg_metrics)

# All extended metrics ranked by out-of-sample deviance
print(extended_metrics[order(extended_metrics$OutOfSample), ], row.names = FALSE)

# ==============================================================================
# PART 2: GAM AS REFERENCE POINT
# ==============================================================================
# Offset inside formula so predict.gam works correctly on new data

time_gam_start <- Sys.time()
gam1 <- gam(ClaimNb ~ s(DrivAge) + s(BonusMalus, k = 20) + s(log(Density)) +
              VehPowerGLM + VehAgeGLM + VehBrand + VehGas + Region +
              offset(log(Exposure)),
            family = poisson(), data = learn,
            method = "REML")
time_gam <- as.numeric(difftime(Sys.time(), time_gam_start, units = "secs"))

summary(gam1)

# GAM with interactions
time_gam2_start <- Sys.time()
gam2 <- gam(ClaimNb ~ s(DrivAge) + s(BonusMalus, k = 20) + s(log(Density)) +
              VehPowerGLM + VehAgeGLM + VehBrand + VehGas + Region +
              VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM +
              offset(log(Exposure)),
            family = poisson(), data = learn,
            method = "REML")
time_gam2 <- as.numeric(difftime(Sys.time(), time_gam2_start, units = "secs"))

gam_evaluate <- function(model, name, learn, test, runtime) {
  pred_learn <- predict(model, type = "response") / learn$Exposure
  pred_test  <- predict(model, newdata = test, type = "response") / test$Exposure
  data.frame(
    Model       = name,
    Runtime     = sprintf("%ds", round(runtime)),
    InSample    = round(poisson_deviance(learn$ClaimNb, pred_learn, learn$Exposure), 5),
    OutOfSample = round(poisson_deviance(test$ClaimNb, pred_test, test$Exposure), 5),
    Gini_test   = round(gini_coefficient(test$ClaimNb, pred_test, test$Exposure), 4),
    Calib_test  = round(calibration_ratio(test$ClaimNb, pred_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}

gam_results <- rbind(
  gam_evaluate(gam1, "GAM1 (smooth main effects)", learn, test, time_gam),
  gam_evaluate(gam2, "GAM2 (smooth + interactions)", learn, test, time_gam2)
)

print(gam_results, row.names = FALSE)

# GAM smooth effect plots
pdf("gam_smooth_effects.pdf", width = 12, height = 5)
par(mfrow = c(1, 3))
plot(gam1, select = 1, shade = TRUE, main = "GAM: Driver Age Effect")
plot(gam1, select = 2, shade = TRUE, main = "GAM: Bonus-Malus Effect")
plot(gam1, select = 3, shade = TRUE, main = "GAM: log(Density) Effect")
dev.off()

# ==============================================================================
# PART 3: REGION GROUPING
# ==============================================================================

region_coefs <- coef(glm8)[grep("Region", names(coef(glm8)))]
region_coefs <- c("RegionR24" = 0, region_coefs)

region_df <- data.frame(
  Region = gsub("Region", "", names(region_coefs)),
  Coefficient = as.numeric(region_coefs),
  stringsAsFactors = FALSE
)
region_df <- region_df[order(region_df$Coefficient), ]

region_df$Group <- cut(region_df$Coefficient,
                       breaks = quantile(region_df$Coefficient, probs = c(0, 0.25, 0.5, 0.75, 1)),
                       include.lowest = TRUE, labels = c("Low", "MedLow", "MedHigh", "High"))

print(region_df, row.names = FALSE)

region_map <- setNames(region_df$Group, region_df$Region)
learn$RegionGroup <- factor(region_map[as.character(learn$Region)],
                            levels = c("Low", "MedLow", "MedHigh", "High"))
test$RegionGroup  <- factor(region_map[as.character(test$Region)],
                            levels = c("Low", "MedLow", "MedHigh", "High"))

# GLM10: grouped regions (based on GLM8 + grouped Region)
time_glm10_start <- Sys.time()
glm10 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM +
               ns(DrivAge, df = 4) + ns(BonusMalusGLM, df = 4) +
               VehBrand + VehGas + DensityGLM + RegionGroup +
               VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM,
             family = poisson(), data = learn, offset = log(Exposure))
time_glm10 <- as.numeric(difftime(Sys.time(), time_glm10_start, units = "secs"))

glm10_result <- evaluate_model(glm10, "GLM10 (grouped regions)", learn, test, time_glm10)
glm8_result  <- evaluate_model(glm8, "GLM8 (optimised)", learn, test)
glm9_result  <- evaluate_model(glm9, "GLM9 (3-way)", learn, test)
print(rbind(glm8_result, glm9_result, glm10_result), row.names = FALSE)

# ==============================================================================
# PART 4: HIGHER-ORDER INTERACTIONS (RECORD OF SEARCH)
# ==============================================================================
# These were already tested — results recorded here for completeness.
# The best 3-way interaction (VehAge:VehPower:VehGas) is now GLM9 in script 04.

base_aic <- AIC(glm8)

three_way_candidates <- c(
  "VehAgeGLM:VehBrand:VehGas",
  "VehAgeGLM:VehPowerGLM:VehGas",
  "VehAgeGLM:VehGas:DensityGLM"
)

three_way_results <- data.frame(
  Interaction = character(), AIC = numeric(), AIC_drop = numeric(),
  n_extra_params = integer(), OutOfSample = numeric(),
  stringsAsFactors = FALSE
)

base_formula_text <- "ClaimNb ~ VehPowerGLM + VehAgeGLM + ns(DrivAge, df = 4) + ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + DensityGLM + Region + VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM"

for (int_term in three_way_candidates) {
  new_formula <- as.formula(paste(base_formula_text, "+", int_term))
  
  fit <- tryCatch(
    glm(new_formula, family = poisson(), data = learn, offset = log(Exposure)),
    error = function(e) NULL
  )
  
  if (!is.null(fit)) {
    pred_test <- predict(fit, newdata = test, type = "response") / test$Exposure
    oos <- poisson_deviance(test$ClaimNb, pred_test, test$Exposure)
    
    three_way_results <- rbind(three_way_results, data.frame(
      Interaction    = int_term,
      AIC            = round(AIC(fit)),
      AIC_drop       = round(base_aic - AIC(fit)),
      n_extra_params = length(coef(fit)) - length(coef(glm8)),
      OutOfSample    = round(oos, 5),
      stringsAsFactors = FALSE
    ))
  }
}

print(three_way_results, row.names = FALSE)

# ==============================================================================
# PART 5: LIFT CHART BY DECILE
# ==============================================================================

create_lift_table <- function(model_name, predicted, actual, exposure) {
  decile <- cut(predicted, breaks = quantile(predicted, probs = seq(0, 1, 0.1)),
                include.lowest = TRUE, labels = 1:10)
  
  lift <- aggregate(
    data.frame(Actual = actual, Predicted = predicted * exposure, Exposure = exposure),
    by = list(Decile = decile), FUN = sum
  )
  lift$Actual_freq    <- lift$Actual / lift$Exposure
  lift$Predicted_freq <- lift$Predicted / lift$Exposure
  lift$Ratio          <- round(lift$Actual_freq / lift$Predicted_freq, 3)
  lift$Model          <- model_name
  lift[, c("Model", "Decile", "Actual_freq", "Predicted_freq", "Ratio")]
}

pred_glm1  <- predict(glm1, newdata = test, type = "response") / test$Exposure
pred_glm8  <- predict(glm8, newdata = test, type = "response") / test$Exposure
pred_glm9  <- predict(glm9, newdata = test, type = "response") / test$Exposure
pred_lasso <- as.numeric(predict(cv_lasso, newx = X_test, s = "lambda.min",
                                  newoffset = offset_test, type = "response")) / test$Exposure

lift_glm1  <- create_lift_table("GLM1", pred_glm1, test$ClaimNb, test$Exposure)
lift_glm8  <- create_lift_table("GLM8", pred_glm8, test$ClaimNb, test$Exposure)
lift_glm9  <- create_lift_table("GLM9", pred_glm9, test$ClaimNb, test$Exposure)
lift_lasso <- create_lift_table("Lasso", pred_lasso, test$ClaimNb, test$Exposure)

print(lift_glm1, row.names = FALSE)
print(lift_glm8, row.names = FALSE)
print(lift_glm9, row.names = FALSE)
print(lift_lasso, row.names = FALSE)

pdf("lift_chart_comparison.pdf", width = 12, height = 10)
par(mfrow = c(2, 2))

barplot(rbind(lift_glm1$Actual_freq, lift_glm1$Predicted_freq),
        beside = TRUE, names.arg = 1:10, col = c("steelblue", "tomato"),
        xlab = "Risk Decile", ylab = "Claim Frequency",
        main = "GLM1: Actual vs Predicted by Decile")
legend("topleft", legend = c("Actual", "Predicted"), fill = c("steelblue", "tomato"), cex = 0.8)

barplot(rbind(lift_glm8$Actual_freq, lift_glm8$Predicted_freq),
        beside = TRUE, names.arg = 1:10, col = c("steelblue", "tomato"),
        xlab = "Risk Decile", ylab = "Claim Frequency",
        main = "GLM8: Actual vs Predicted by Decile")
legend("topleft", legend = c("Actual", "Predicted"), fill = c("steelblue", "tomato"), cex = 0.8)

barplot(rbind(lift_glm9$Actual_freq, lift_glm9$Predicted_freq),
        beside = TRUE, names.arg = 1:10, col = c("steelblue", "tomato"),
        xlab = "Risk Decile", ylab = "Claim Frequency",
        main = "GLM9: Actual vs Predicted by Decile")
legend("topleft", legend = c("Actual", "Predicted"), fill = c("steelblue", "tomato"), cex = 0.8)

barplot(rbind(lift_lasso$Actual_freq, lift_lasso$Predicted_freq),
        beside = TRUE, names.arg = 1:10, col = c("steelblue", "tomato"),
        xlab = "Risk Decile", ylab = "Claim Frequency",
        main = "Lasso: Actual vs Predicted by Decile")
legend("topleft", legend = c("Actual", "Predicted"), fill = c("steelblue", "tomato"), cex = 0.8)

dev.off()

# ==============================================================================
# PART 6: COMPREHENSIVE COMPARISON TABLE
# ==============================================================================

# --- GLM rows (includes GLM9) ---
glm_rows <- do.call(rbind, lapply(names(all_glm_models), function(nm) {
  mod <- all_glm_models[[nm]]
  pred_test <- predict(mod, newdata = test, type = "response") / test$Exposure
  data.frame(
    Model = nm, Type = "GLM",
    OutOfSample = round(poisson_deviance(test$ClaimNb, pred_test, test$Exposure), 5),
    Gini_test = round(gini_coefficient(test$ClaimNb, pred_test, test$Exposure), 4),
    Calib_test = round(calibration_ratio(test$ClaimNb, pred_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

# --- GLM10 row ---
pred_glm10 <- predict(glm10, newdata = test, type = "response") / test$Exposure
glm10_comp_row <- data.frame(
  Model = "GLM10 (grouped regions)", Type = "GLM",
  OutOfSample = round(poisson_deviance(test$ClaimNb, pred_glm10, test$Exposure), 5),
  Gini_test = round(gini_coefficient(test$ClaimNb, pred_glm10, test$Exposure), 4),
  Calib_test = round(calibration_ratio(test$ClaimNb, pred_glm10, test$Exposure), 4),
  stringsAsFactors = FALSE
)

# --- Regularised rows ---
reg_comp_rows <- do.call(rbind, lapply(names(reg_model_list), function(nm) {
  cv_mod <- reg_model_list[[nm]]$cv
  s_val  <- reg_model_list[[nm]]$s
  
  pred_test <- as.numeric(predict(cv_mod, newx = X_test, s = s_val,
                                  newoffset = offset_test, type = "response"))
  mu_test <- pred_test / test$Exposure
  
  data.frame(
    Model = nm, Type = "Regularised",
    OutOfSample = round(poisson_deviance(test$ClaimNb, mu_test, test$Exposure), 5),
    Gini_test = round(gini_coefficient(test$ClaimNb, mu_test, test$Exposure), 4),
    Calib_test = round(calibration_ratio(test$ClaimNb, mu_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

# --- GAM rows ---
gam_comp_rows <- do.call(rbind, lapply(list(
  list(gam1, "GAM1 (smooth)"),
  list(gam2, "GAM2 (smooth+interact)")
), function(x) {
  mod <- x[[1]]; nm <- x[[2]]
  pred_test <- predict(mod, newdata = test, type = "response") / test$Exposure
  data.frame(
    Model = nm, Type = "GAM",
    OutOfSample = round(poisson_deviance(test$ClaimNb, pred_test, test$Exposure), 5),
    Gini_test = round(gini_coefficient(test$ClaimNb, pred_test, test$Exposure), 4),
    Calib_test = round(calibration_ratio(test$ClaimNb, pred_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

# --- Tree rows ---
tree_models <- list(
  list(tree1, "RT1"),
  list(tree2, "RT2 (min CV)"),
  list(tree3, "RT3 (1-SD)"),
  list(tree1000, "RT_1000")
)

tree_comp_rows <- do.call(rbind, lapply(tree_models, function(x) {
  tr <- x[[1]]; nm <- x[[2]]
  pred_test <- predict(tr, newdata = test)
  data.frame(
    Model = nm, Type = "Tree",
    OutOfSample = round(poisson_deviance(test$ClaimNb, pred_test, test$Exposure), 5),
    Gini_test = round(gini_coefficient(test$ClaimNb, pred_test, test$Exposure), 4),
    Calib_test = round(calibration_ratio(test$ClaimNb, pred_test, test$Exposure), 4),
    stringsAsFactors = FALSE
  )
}))

# --- Combine everything ---
full_comparison <- rbind(glm_rows, glm10_comp_row, reg_comp_rows, gam_comp_rows, tree_comp_rows)
full_comparison <- full_comparison[order(full_comparison$OutOfSample), ]

# Master comparison table
print(full_comparison, row.names = FALSE)

# ==============================================================================
# SAVE
# ==============================================================================

save(gam1, gam2, glm10, region_df,
     extended_metrics, gam_results, three_way_results,
     lift_glm1, lift_glm8, lift_glm9, lift_lasso, full_comparison,
     file = "further_improvements.RData")

write.csv(full_comparison, "full_model_comparison.csv", row.names = FALSE)
write.csv(extended_metrics, "extended_metrics_all.csv", row.names = FALSE)
write.csv(region_df, "region_grouping.csv", row.names = FALSE)
