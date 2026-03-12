# ==============================================================================
# 00_utils.R — Shared functions for all scripts
# ==============================================================================

# Poisson deviance loss (as per Section 2.2 of Noll et al.)
# y = observed counts, mu = predicted rates, w = exposures
# Returns average deviance loss in units of 10^-2
poisson_deviance <- function(y, mu, w) {
  idx_zero <- (y == 0)
  dev <- numeric(length(y))
  dev[idx_zero]  <- 2 * mu[idx_zero] * w[idx_zero]
  dev[!idx_zero] <- 2 * y[!idx_zero] * 
    (log(y[!idx_zero] / (mu[!idx_zero] * w[!idx_zero]))) + 
    2 * (mu[!idx_zero] * w[!idx_zero] - y[!idx_zero])
  mean(dev) * 100
}

# Helper to evaluate a model on learn/test and return a one-row data frame
evaluate_model <- function(model, model_name, learn, test, runtime = NA) {
  learn_pred <- predict(model, type = "response") / learn$Exposure
  test_pred  <- predict(model, newdata = test, type = "response") / test$Exposure
  
  data.frame(
    Model       = model_name,
    Runtime     = ifelse(is.na(runtime), "–", sprintf("%ds", round(runtime))),
    Parameters  = length(coef(model)),
    AIC         = round(AIC(model)),
    InSample    = round(poisson_deviance(learn$ClaimNb, learn_pred, learn$Exposure), 5),
    OutOfSample = round(poisson_deviance(test$ClaimNb, test_pred, test$Exposure), 5),
    stringsAsFactors = FALSE
  )
}
