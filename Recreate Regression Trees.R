# ==============================================================================
# Regression Tree Models - TABLE 6
# French Motor Third-Party Liability Claims
# Section 4: Regression Trees (Pages 18-24)
# ==============================================================================
setwd("C:/Users/Frank/OneDrive - University College Cork/4.2/ST 4050 - Stats 2")
# Load the saved workspace from the GLM script
load("frequency_models.RData")

library(rpart)
library(rpart.plot)

# Poisson deviance loss function
poisson_deviance <- function(y, mu, w) {
  idx_zero <- (y == 0)
  dev <- numeric(length(y))
  dev[idx_zero] <- 2 * mu[idx_zero] * w[idx_zero]
  dev[!idx_zero] <- 2 * y[!idx_zero] * 
    (log(y[!idx_zero]/(mu[!idx_zero] * w[!idx_zero]))) + 
    2 * (mu[!idx_zero] * w[!idx_zero] - y[!idx_zero])
  mean(dev) * 100
}

# ==============================================================================
# MODEL RT1: Tree with cp=0.0005, minbucket=10000 (Page 20, Figure 14)
# ==============================================================================
time_rt1_start <- Sys.time()

tree1 <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                 BonusMalus + VehBrand + VehGas + Density + Region,
               data = learn,
               method = "poisson",
               control = rpart.control(xval = 1, minbucket = 10000, cp = 0.0005))

time_rt1 <- as.numeric(difftime(Sys.time(), time_rt1_start, units = "secs"))

# Display tree structure
print(tree1)

# Plot the tree
pdf("tree_rt1.pdf", width = 14, height = 10)
rpart.plot(tree1, type = 0, extra = 101,cex =0.7, under = FALSE, fallen.leaves = TRUE,
           main = "Model RT1: Regression Tree (K=12 leaves)")
dev.off()

# ==============================================================================
# MODEL RT2: Minimal CV rule (Page 23, Figure 15)
# ==============================================================================

time_rt2_start <- Sys.time()

# Build a large tree first
tree2 <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                 BonusMalus + VehBrand + VehGas + Density + Region,
               data = learn,
               method = "poisson",
               control = rpart.control(xval = 10, minbucket = 10000, 
                                       cp = 0.00001))

# Find the optimal tree using minimal CV rule
cp_opt <- tree2$cptable[which.min(tree2$cptable[, "xerror"]), "CP"]
tree2_pruned <- prune(tree2, cp = cp_opt)

time_rt2 <- as.numeric(difftime(Sys.time(), time_rt2_start, units = "secs"))

cp_opt
nrow(tree2_pruned$frame[tree2_pruned$frame$var == "<leaf>", ])

# Plot cross-validation results
pdf("/mnt/user-data/outputs/tree_rt2_cv.pdf", width = 10, height = 7)
plotcp(tree2, main = "Cross-Validation Results for Regression Tree")
dev.off()

# ==============================================================================
# MODEL RT3: 1-SD rule (Page 23, Figure 15 bottom)
# ==============================================================================

time_rt3_start <- Sys.time()

# Find CP for 1-SD rule
xerr <- tree2$cptable[, "xerror"]
xstd <- tree2$cptable[, "xstd"]
min_xerr <- min(xerr)
threshold <- min_xerr + xstd[which.min(xerr)]

# Find simplest model within 1 SD
idx_1sd <- min(which(xerr <= threshold))
cp_1sd <- tree2$cptable[idx_1sd, "CP"]
tree3 <- prune(tree2, cp = cp_1sd)

time_rt3 <- time_rt2  # Same fitting time, different pruning

cp_1sd
nrow(tree3$frame[tree3$frame$var == "<leaf>", ])

# Plot the 1-SD rule tree
pdf("/mnt/user-data/outputs/tree_rt3.pdf", width = 12, height = 8)
rpart.plot(tree3, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE,
           main = "Model RT3: 1-SD Rule Tree (K=6 leaves)")
dev.off()

# ==============================================================================
# MODEL RT_1000: Minimal size 1000 (Page 24, Table 7)
# ==============================================================================

time_rt1000_start <- Sys.time()

tree1000 <- rpart(cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
                    BonusMalus + VehBrand + VehGas + Density + Region,
                  data = learn,
                  method = "poisson",
                  control = rpart.control(xval = 10, minbucket = 1000, 
                                          cp = 0.00001))

# Prune using minimal CV rule
cp_opt_1000 <- tree1000$cptable[which.min(tree1000$cptable[, "xerror"]), "CP"]
tree1000_pruned <- prune(tree1000, cp = cp_opt_1000)

time_rt1000 <- as.numeric(difftime(Sys.time(), time_rt1000_start, units = "secs"))

cat("Number of leaves:", 
    nrow(tree1000_pruned$frame[tree1000_pruned$frame$var == "<leaf>", ]), "\n\n")

# ==============================================================================
# CALCULATE LOSSES FOR ALL TREE MODELS
# ==============================================================================

# Model RT1
learn$pred_rt1 <- predict(tree1, newdata = learn) / learn$Exposure
test$pred_rt1 <- predict(tree1, newdata = test) / test$Exposure
loss_rt1_in <- poisson_deviance(learn$ClaimNb, learn$pred_rt1, learn$Exposure)
loss_rt1_out <- poisson_deviance(test$ClaimNb, test$pred_rt1, test$Exposure)

# Model RT2
learn$pred_rt2 <- predict(tree2_pruned, newdata = learn) / learn$Exposure
test$pred_rt2 <- predict(tree2_pruned, newdata = test) / test$Exposure
loss_rt2_in <- poisson_deviance(learn$ClaimNb, learn$pred_rt2, learn$Exposure)
loss_rt2_out <- poisson_deviance(test$ClaimNb, test$pred_rt2, test$Exposure)

# Model RT3
learn$pred_rt3 <- predict(tree3, newdata = learn) / learn$Exposure
test$pred_rt3 <- predict(tree3, newdata = test) / test$Exposure
loss_rt3_in <- poisson_deviance(learn$ClaimNb, learn$pred_rt3, learn$Exposure)
loss_rt3_out <- poisson_deviance(test$ClaimNb, test$pred_rt3, test$Exposure)

# Model RT_1000
learn$pred_rt1000 <- predict(tree1000_pruned, newdata = learn) / learn$Exposure
test$pred_rt1000 <- predict(tree1000_pruned, newdata = test) / test$Exposure
loss_rt1000_in <- poisson_deviance(learn$ClaimNb, learn$pred_rt1000, learn$Exposure)
loss_rt1000_out <- poisson_deviance(test$ClaimNb, test$pred_rt1000, test$Exposure)

# Get number of parameters (leaves)
n_leaves_rt1 <- nrow(tree1$frame[tree1$frame$var == "<leaf>", ])
n_leaves_rt2 <- nrow(tree2_pruned$frame[tree2_pruned$frame$var == "<leaf>", ])
n_leaves_rt3 <- nrow(tree3$frame[tree3$frame$var == "<leaf>", ])
n_leaves_rt1000 <- nrow(tree1000_pruned$frame[tree1000_pruned$frame$var == "<leaf>", ])

# ==============================================================================
# CREATE TABLE 6: Regression Tree Results (Page 23)
# ==============================================================================

# Compare with homogeneous and GLM1 from previous results
lambda_homog <- sum(learn$ClaimNb) / sum(learn$Exposure)
loss_homog_in <- poisson_deviance(learn$ClaimNb, rep(lambda_homog, nrow(learn)), 
                                  learn$Exposure)
loss_homog_out <- poisson_deviance(test$ClaimNb, rep(lambda_homog, nrow(test)), 
                                   test$Exposure)

# Get GLM1 losses (from previous script)
learn$pred_glm1 <- predict(glm1, newdata = learn, type = "response") / learn$Exposure
test$pred_glm1 <- predict(glm1, newdata = test, type = "response") / test$Exposure
loss_glm1_in <- poisson_deviance(learn$ClaimNb, learn$pred_glm1, learn$Exposure)
loss_glm1_out <- poisson_deviance(test$ClaimNb, test$pred_glm1, test$Exposure)

# Create results table
table6 <- data.frame(
  Model = c("Homogeneous model", "Model RT1", "Model RT2 (min. CV rule)", 
            "Model RT3 (1-SD rule)", "Model GLM1"),
  Runtime = c("–", sprintf("%ds", round(time_rt1)), 
              sprintf("%ds", round(time_rt2)), 
              sprintf("%ds", round(time_rt3)), 
              sprintf("%ds", round(time_glm1))),
  Parameters = c(1, n_leaves_rt1, n_leaves_rt2, n_leaves_rt3, 
                 length(coef(glm1))),
  InSample = sprintf("%.5f", c(loss_homog_in, loss_rt1_in, loss_rt2_in, 
                               loss_rt3_in, loss_glm1_in)),
  OutOfSample = sprintf("%.5f", c(loss_homog_out, loss_rt1_out, loss_rt2_out, 
                                  loss_rt3_out, loss_glm1_out))
)

colnames(table6) <- c("Model", "Run time", "# param.", "In-sample loss", 
                      "Out-of-sample loss")

print(table6, row.names = FALSE)

# ==============================================================================
# CREATE TABLE 7: Results with minbucket=1000 (Page 24)
# ==============================================================================

table7 <- data.frame(
  Model = c("Homogeneous model", "Model RT_1000", "Model GLM1"),
  Runtime = c("–", sprintf("%ds", round(time_rt1000)), 
              sprintf("%ds", round(time_glm1))),
  Parameters = c(1, n_leaves_rt1000, length(coef(glm1))),
  InSample = sprintf("%.5f", c(loss_homog_in, loss_rt1000_in, loss_glm1_in)),
  OutOfSample = sprintf("%.5f", c(loss_homog_out, loss_rt1000_out, loss_glm1_out))
)

colnames(table7) <- c("Model", "Run time", "# param.", "In-sample loss", 
                      "Out-of-sample loss")

print(table7, row.names = FALSE)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

# Save workspace
save(tree1, tree2_pruned, tree3, tree1000_pruned, table6, table7,
     file = "/mnt/user-data/outputs/regression_tree_models.RData",
     compress = TRUE)

# Save tables as CSV
write.csv(table6, "/mnt/user-data/outputs/table6_tree_results.csv", 
          row.names = FALSE)
write.csv(table7, "/mnt/user-data/outputs/table7_tree_results_1000.csv", 
          row.names = FALSE)
