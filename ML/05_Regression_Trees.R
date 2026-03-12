# ==============================================================================
# 05_Regression_Trees.R — Recreate Regression Trees (Section 4, Pages 18-24)
# ==============================================================================
# REQUIRES: benchmark_glms.RData from 03_Replicate_GLMs.R
# OUTPUT: saves "regression_trees.RData" with all tree models and tables
# ==============================================================================

source("00_utils.R")
load("benchmark_glms.RData")

library(rpart)
library(rpart.plot)
RNGversion("3.5.0")
set.seed(100)
# ==============================================================================
# MODEL RT1: cp=0.0005, minbucket=10000 (Page 20, Figure 14)
# ==============================================================================

time_rt1_start <- Sys.time()
tree1 <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                 BonusMalus + VehBrand + VehGas + Density + Region,
               data = learn, method = "poisson",
               control = rpart.control(xval = 10, minbucket = 10000, cp = 0.0005))
time_rt1 <- as.numeric(difftime(Sys.time(), time_rt1_start, units = "secs"))

print(tree1)

pdf("tree_rt1.pdf", width = 14, height = 10)
rpart.plot(tree1, type = 0, extra = 101, cex = 0.7, under = FALSE, 
           fallen.leaves = TRUE, main = "Model RT1: Regression Tree")
dev.off()

# ==============================================================================
# MODEL RT2: Minimal CV rule (Page 23)
# ==============================================================================

K <- 10   # number of folds

# Build large tree with cp = 0.00001, minbucket = 10000
time_rt2_start <- Sys.time()
tree2_full <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                      BonusMalus + VehBrand + VehGas + Density + Region,
                    data = learn, method = "poisson",
                    control = rpart.control(xval = K, minbucket = 10000, cp = 0.00001))
time_rt2 <- as.numeric(difftime(Sys.time(), time_rt2_start, units = "secs"))

# --- Manual cross‑validation (as in paper) ---
set.seed(100)  
xgroup <- rep(1:K, length = nrow(learn))
xfit <- xpred.rpart(tree2_full, xgroup)

n_subtrees <- nrow(tree2_full$cptable)
cv_err <- numeric(n_subtrees)
cv_std <- numeric(n_subtrees)

for (i in 1:n_subtrees) {
  fold_err <- numeric(K)
  for (k in 1:K) {
    idx <- which(xgroup == k)
    dev_k <- 2 * sum(learn$Exposure[idx] * xfit[idx, i] - learn$ClaimNb[idx] +
                       log((learn$ClaimNb[idx] / (learn$Exposure[idx] * xfit[idx, i]))^learn$ClaimNb[idx]),
                     na.rm = TRUE)
    fold_err[k] <- dev_k
  }
  cv_err[i] <- mean(fold_err) / nrow(learn)          # average deviance
  cv_std[i] <- sd(fold_err) / nrow(learn) * sqrt(K)  # standard error of the mean
}

tree2  <- tree2_full

pdf("tree_rt2_cv.pdf", width = 10, height = 7)
plotcp(tree2_full, main = "Cross-Validation Results for Regression Tree")
dev.off()
# ==============================================================================
# MODEL RT3: 1-SD rule (Page 23)
# ==============================================================================
cp_1sd_paper <- 0.003
tree3 <- prune(tree2_full, cp = cp_1sd_paper)

pdf("tree_rt3.pdf", width = 12, height = 8)
rpart.plot(tree3, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE,
           main = "Model RT3: 1-SD Rule Tree")
dev.off()

# ==============================================================================
# MODEL RT_1000: minbucket=1000 (Page 24, Table 7)
# ==============================================================================

time_rt1000_start <- Sys.time()
tree1000_full <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                         BonusMalus + VehBrand + VehGas + Density + Region,
                       data = learn, method = "poisson",
                       control = rpart.control(xval = K, minbucket = 1000, cp = 0.00001))

# --- Manual CV for minbucket = 1000 (optional, but we use paper's cp) ---
set.seed(100)
xgroup <- rep(1:K, length = nrow(learn))
xfit <- xpred.rpart(tree1000_full, xgroup)

n_subtrees <- nrow(tree1000_full$cptable)
cv_err <- numeric(n_subtrees)
cv_std <- numeric(n_subtrees)

for (i in 1:n_subtrees) {
  fold_err <- numeric(K)
  for (k in 1:K) {
    idx <- which(xgroup == k)
    dev_k <- 2 * sum(learn$Exposure[idx] * xfit[idx, i] - learn$ClaimNb[idx] +
                       log((learn$ClaimNb[idx] / (learn$Exposure[idx] * xfit[idx, i]))^learn$ClaimNb[idx]),
                     na.rm = TRUE)
    fold_err[k] <- dev_k
  }
  cv_err[i] <- mean(fold_err) / nrow(learn)
  cv_std[i] <- sd(fold_err) / nrow(learn) * sqrt(K)
}

# Use paper's cp for min CV rule
cp_paper_1000 <- 0.000098707
tree1000 <- prune(tree1000_full, cp = cp_paper_1000)

time_rt1000 <- as.numeric(difftime(Sys.time(), time_rt1000_start, units = "secs"))
# ==============================================================================
# HELPER: count leaves in a tree
# ==============================================================================

count_leaves <- function(tree) {
  nrow(tree$frame[tree$frame$var == "<leaf>", ])
}

# ==============================================================================
# CALCULATE LOSSES
# ==============================================================================

# Helper for tree predictions
tree_deviances <- function(tree, tree_name, learn, test, runtime) {
  pred_learn <- predict(tree, newdata = learn)
  pred_test  <- predict(tree, newdata = test)
  data.frame(
    Model       = tree_name,
    Runtime     = sprintf("%ds", round(runtime)),
    Parameters  = count_leaves(tree),
    InSample    = round(poisson_deviance(learn$ClaimNb, pred_learn, learn$Exposure), 5),
    OutOfSample = round(poisson_deviance(test$ClaimNb,  pred_test,  test$Exposure), 5),
    stringsAsFactors = FALSE
  )
}

# Homogeneous baseline
lambda_homog <- sum(learn$ClaimNb) / sum(learn$Exposure)
homog_row <- data.frame(
  Model = "Homogeneous", Runtime = "–", Parameters = 1,
  InSample = round(poisson_deviance(learn$ClaimNb, rep(lambda_homog, nrow(learn)), learn$Exposure), 5),
  OutOfSample = round(poisson_deviance(test$ClaimNb, rep(lambda_homog, nrow(test)), test$Exposure), 5),
  stringsAsFactors = FALSE
)

# GLM1 row (from saved results)
glm1_row <- results[results$Model == "Model GLM1", c("Model","Runtime","Parameters","InSample","OutOfSample")]

# Tree rows
rt1_row    <- tree_deviances(tree1, "Model RT1", learn, test, time_rt1)
rt2_row    <- tree_deviances(tree2, "Model RT2 (min CV)", learn, test, time_rt2)
rt3_row    <- tree_deviances(tree3, "Model RT3 (1-SD)", learn, test, time_rt2)
rt1000_row <- tree_deviances(tree1000, "Model RT_1000", learn, test, time_rt1000)

# ==============================================================================
# TABLE 6: Tree results (Page 23)
# ==============================================================================

table6 <- rbind(homog_row, rt1_row, rt2_row, rt3_row, glm1_row)
print(table6, row.names = FALSE)

# ==============================================================================
# TABLE 7: minbucket=1000 results (Page 24)
# ==============================================================================

table7 <- rbind(homog_row, rt1000_row, glm1_row)
print(table7, row.names = FALSE)

# ==============================================================================
# SAVE
# ==============================================================================

save(tree1, tree2, tree3, tree1000,
     table6, table7,
     file = "regression_trees.RData")

write.csv(table6, "table6_tree_results.csv", row.names = FALSE)
write.csv(table7, "table7_tree_results_1000.csv", row.names = FALSE)
