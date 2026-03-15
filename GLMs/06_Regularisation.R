# ==============================================================================
# 06_Regularisation.R — Ridge, Lasso, Elastic Net + Overfitting Diagnostics
# ==============================================================================
# REQUIRES: improved_glms.RData from 04_Improve_GLMs.R
# OUTPUT: saves "regularisation_results.RData"
# ==============================================================================

source("00_utils.R")
load("improved_glms.RData")

if (!require("glmnet")) install.packages("glmnet")
library(glmnet)
library(splines)

# ==============================================================================
# PART 1: OVERFITTING DIAGNOSTICS
# ==============================================================================

all_models <- list(
  "Homogeneous"             = glm_homog,
  "GLM1 (benchmark)"        = glm1,
  "GLM2 (drop Area)"        = glm2,
  "GLM3 (drop Area+Brand)"  = glm3,
  "GLM4 (interactions)"     = glm4,
  "GLM5 (splines)"          = glm5,
  "GLM6 (splines+interact)" = glm6,
  "GLM7 (polynomials)"      = glm7,
  "GLM8 (optimised)"        = glm8,
  "GLM9 (3-way)"            = glm9
)

overfit_diag <- do.call(rbind, lapply(names(all_models), function(nm) {
  mod <- all_models[[nm]]
  learn_pred <- predict(mod, type = "response") / learn$Exposure
  test_pred  <- predict(mod, newdata = test, type = "response") / test$Exposure
  in_loss  <- poisson_deviance(learn$ClaimNb, learn_pred, learn$Exposure)
  out_loss <- poisson_deviance(test$ClaimNb, test_pred, test$Exposure)
  data.frame(
    Model      = nm,
    n_params   = length(coef(mod)),
    in_sample  = round(in_loss, 5),
    out_sample = round(out_loss, 5),
    gap        = round(out_loss - in_loss, 5),
    AIC        = round(AIC(mod)),
    BIC        = round(BIC(mod)),
    stringsAsFactors = FALSE
  )
}))

print(overfit_diag, row.names = FALSE)

# --- Overfitting diagnostic plots ---

pdf("overfitting_diagnostic.pdf", width = 10, height = 6)
par(mfrow = c(1, 2), mar = c(8, 4, 3, 1))

model_ids <- 1:nrow(overfit_diag)
plot(model_ids, overfit_diag$in_sample, type = "b", pch = 19, col = "blue",
     ylim = range(c(overfit_diag$in_sample, overfit_diag$out_sample)),
     xlab = "", ylab = "Poisson Deviance (x10^-2)",
     main = "In-Sample vs Out-of-Sample Deviance", xaxt = "n")
lines(model_ids, overfit_diag$out_sample, type = "b", pch = 17, col = "red")
axis(1, at = model_ids, labels = overfit_diag$Model, las = 2, cex.axis = 0.7)
legend("topright", legend = c("In-sample", "Out-of-sample"),
       col = c("blue", "red"), pch = c(19, 17), lty = 1, cex = 0.8)

plot(overfit_diag$n_params, overfit_diag$gap, pch = 19, col = "darkred",
     xlab = "Number of Parameters", ylab = "Deviance Gap (out - in)",
     main = "Overfitting Gap vs Model Complexity")
text(overfit_diag$n_params, overfit_diag$gap, labels = overfit_diag$Model,
     pos = 3, cex = 0.6)
abline(h = 0, lty = 2, col = "grey")

dev.off()

# ==============================================================================
# PART 2: 5-FOLD CROSS-VALIDATION ON UNREGULARISED MODELS
# ==============================================================================

K <- 5
fold_ids <- sample(rep(1:K, length.out = nrow(learn)))

cv_formulas <- list(
  "GLM1 (benchmark)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + 
    BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM,
  
  "GLM4 (interactions)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + 
    BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM +
    VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM,
  
  "GLM5 (splines)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + ns(DrivAge, df = 4) + 
    ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + DensityGLM + Region + AreaGLM,
  
  "GLM6 (splines+interact)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + 
    ns(DrivAge, df = 4) + ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + 
    DensityGLM + Region + AreaGLM + VehAgeGLM:VehBrand + VehAgeGLM:VehGas +
    VehPowerGLM:VehAgeGLM,
  
  "GLM8 (optimised)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + ns(DrivAge, df = 4) + 
    ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + DensityGLM + Region +
    VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM,
  
  "GLM9 (3-way)" = ClaimNb ~ VehPowerGLM + VehAgeGLM + ns(DrivAge, df = 4) + 
    ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + DensityGLM + Region +
    VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM +
    VehAgeGLM:VehPowerGLM:VehGas
)

cv_results <- do.call(rbind, lapply(names(cv_formulas), function(model_nm) {
  fold_deviances <- numeric(K)
  for (k in 1:K) {
    train_k <- learn[fold_ids != k, ]
    valid_k <- learn[fold_ids == k, ]
    fit_k   <- glm(cv_formulas[[model_nm]], family = poisson(), 
                   data = train_k, offset = log(Exposure))
    pred_k  <- predict(fit_k, newdata = valid_k, type = "response") / valid_k$Exposure
    fold_deviances[k] <- poisson_deviance(valid_k$ClaimNb, pred_k, valid_k$Exposure)
  }
  data.frame(Model = model_nm, CV_mean = round(mean(fold_deviances), 5),
             CV_sd = round(sd(fold_deviances), 5), stringsAsFactors = FALSE)
}))

print(cv_results, row.names = FALSE)

# ==============================================================================
# PART 3: REGULARISED GLMs WITH glmnet
# ==============================================================================
# Uses GLM9 formula (richest feature set including 3-way interaction)

design_formula <- ~ VehPowerGLM + VehAgeGLM + ns(DrivAge, df = 4) + 
  ns(BonusMalusGLM, df = 4) + VehBrand + VehGas + DensityGLM + Region + AreaGLM +
  VehAgeGLM:VehBrand + VehAgeGLM:VehGas + VehPowerGLM:VehAgeGLM +
  VehAgeGLM:VehPowerGLM:VehGas

X_learn <- model.matrix(design_formula, data = learn)[, -1]
X_test  <- model.matrix(design_formula, data = test)[, -1]

y_learn      <- learn$ClaimNb
y_test       <- test$ClaimNb
offset_learn <- log(learn$Exposure)
offset_test  <- log(test$Exposure)

# --- Ridge (alpha = 0) ---
set.seed(100)
cv_ridge <- cv.glmnet(x = X_learn, y = y_learn, family = "poisson",
                       offset = offset_learn, alpha = 0, nfolds = 5,
                       type.measure = "deviance")

# --- Lasso (alpha = 1) ---
set.seed(100)
cv_lasso <- cv.glmnet(x = X_learn, y = y_learn, family = "poisson",
                       offset = offset_learn, alpha = 1, nfolds = 5,
                       type.measure = "deviance")

lasso_coefs_min <- coef(cv_lasso, s = "lambda.min")
lasso_coefs_1se <- coef(cv_lasso, s = "lambda.1se")

# --- Elastic Net (alpha = 0.5) ---
set.seed(100)
cv_enet <- cv.glmnet(x = X_learn, y = y_learn, family = "poisson",
                      offset = offset_learn, alpha = 0.5, nfolds = 5,
                      type.measure = "deviance")

# --- Alpha search ---
alphas <- seq(0, 1, by = 0.1)
alpha_results <- data.frame(alpha = alphas, min_cv_deviance = NA)

set.seed(100)
for (i in seq_along(alphas)) {
  cv_fit <- cv.glmnet(x = X_learn, y = y_learn, family = "poisson",
                       offset = offset_learn, alpha = alphas[i], nfolds = 5,
                       type.measure = "deviance")
  alpha_results$min_cv_deviance[i] <- min(cv_fit$cvm)
}

best_alpha <- alpha_results$alpha[which.min(alpha_results$min_cv_deviance)]
print(alpha_results)

# Refit with best alpha
set.seed(100)
cv_best <- cv.glmnet(x = X_learn, y = y_learn, family = "poisson",
                      offset = offset_learn, alpha = best_alpha, nfolds = 5,
                      type.measure = "deviance")

# ==============================================================================
# PART 4: COMPARE ALL MODELS
# ==============================================================================

reg_models <- list(
  "Ridge (lambda.min)"       = list(cv = cv_ridge, s = "lambda.min"),
  "Ridge (lambda.1se)"       = list(cv = cv_ridge, s = "lambda.1se"),
  "Lasso (lambda.min)"       = list(cv = cv_lasso, s = "lambda.min"),
  "Lasso (lambda.1se)"       = list(cv = cv_lasso, s = "lambda.1se"),
  "Elastic Net (lambda.min)" = list(cv = cv_enet,  s = "lambda.min"),
  "Elastic Net (lambda.1se)" = list(cv = cv_enet,  s = "lambda.1se"),
  "Best alpha (lambda.min)"  = list(cv = cv_best,  s = "lambda.min"),
  "Best alpha (lambda.1se)"  = list(cv = cv_best,  s = "lambda.1se")
)

reg_results <- do.call(rbind, lapply(names(reg_models), function(nm) {
  cv_mod <- reg_models[[nm]]$cv
  s_val  <- reg_models[[nm]]$s
  lam    <- ifelse(s_val == "lambda.min", cv_mod$lambda.min, cv_mod$lambda.1se)
  
  pred_learn <- as.numeric(predict(cv_mod, newx = X_learn, s = s_val,
                                   newoffset = offset_learn, type = "response"))
  pred_test  <- as.numeric(predict(cv_mod, newx = X_test, s = s_val,
                                   newoffset = offset_test, type = "response"))
  
  mu_learn <- pred_learn / learn$Exposure
  mu_test  <- pred_test  / test$Exposure
  
  in_loss  <- poisson_deviance(y_learn, mu_learn, learn$Exposure)
  out_loss <- poisson_deviance(y_test,  mu_test,  test$Exposure)
  
  coefs <- coef(cv_mod, s = s_val)
  n_nz  <- sum(coefs != 0) - 1
  
  data.frame(Model = nm, lambda = round(lam, 6), n_nonzero = n_nz,
             in_sample = round(in_loss, 5), out_sample = round(out_loss, 5),
             gap = round(out_loss - in_loss, 5), stringsAsFactors = FALSE)
}))

print(reg_results, row.names = FALSE)

# --- Combined: unregularised vs regularised ---
combined <- rbind(
  data.frame(Model = overfit_diag$Model, Type = "Unregularised",
             n_params = overfit_diag$n_params, in_sample = overfit_diag$in_sample,
             out_sample = overfit_diag$out_sample, gap = overfit_diag$gap,
             stringsAsFactors = FALSE),
  data.frame(Model = reg_results$Model, Type = "Regularised",
             n_params = reg_results$n_nonzero, in_sample = reg_results$in_sample,
             out_sample = reg_results$out_sample, gap = reg_results$gap,
             stringsAsFactors = FALSE)
)
combined <- combined[order(combined$out_sample), ]
print(combined, row.names = FALSE)

# ==============================================================================
# PART 5: VISUALISATIONS
# ==============================================================================

pdf("regularisation_results.pdf", width = 14, height = 10)
par(mfrow = c(2, 2))

plot(cv_ridge, main = "Ridge: CV Deviance vs log(lambda)")
plot(cv_lasso, main = "Lasso: CV Deviance vs log(lambda)")
plot(cv_enet,  main = "Elastic Net (alpha=0.5): CV Deviance vs log(lambda)")

plot(alpha_results$alpha, alpha_results$min_cv_deviance,
     type = "b", pch = 19, col = "darkblue",
     xlab = "Alpha (0=Ridge, 1=Lasso)", ylab = "Minimum CV Deviance",
     main = "Optimal Alpha Search")
abline(v = best_alpha, lty = 2, col = "red")
text(best_alpha, min(alpha_results$min_cv_deviance),
     sprintf("Best: alpha=%.1f", best_alpha), pos = 4, col = "red")

dev.off()

# Lasso coefficient paths
pdf("lasso_coefficient_path.pdf", width = 10, height = 6)
plot(cv_lasso$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "blue")
abline(v = log(cv_lasso$lambda.1se), lty = 2, col = "red")
legend("topright", legend = c("lambda.min", "lambda.1se"),
       col = c("blue", "red"), lty = 2, cex = 0.8)
dev.off()

# ==============================================================================
# PART 6: LASSO VARIABLE SELECTION
# ==============================================================================

lasso_coefs <- coef(cv_lasso, s = "lambda.1se")
coef_df <- data.frame(
  Variable    = rownames(lasso_coefs),
  Coefficient = as.numeric(lasso_coefs),
  stringsAsFactors = FALSE
)

# Variables kept (non-zero), sorted by magnitude
kept <- coef_df[coef_df$Coefficient != 0, ]
kept <- kept[order(-abs(kept$Coefficient)), ]
print(kept, row.names = FALSE)

# Variables dropped
dropped <- coef_df[coef_df$Coefficient == 0, ]
print(dropped, row.names = FALSE)

# ==============================================================================
# SAVE
# ==============================================================================

save(cv_ridge, cv_lasso, cv_enet, cv_best, best_alpha,
     overfit_diag, cv_results, reg_results, combined,
     X_learn, X_test, offset_learn, offset_test,
     file = "regularisation_results.RData")

write.csv(combined, "model_comparison_all.csv", row.names = FALSE)
