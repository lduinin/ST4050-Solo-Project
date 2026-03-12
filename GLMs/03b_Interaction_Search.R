# ==============================================================================
# 03b_Interaction_Search.R — Systematic search for useful interactions
# ==============================================================================
# REQUIRES: benchmark_glms.RData from 03_Replicate_GLMs.R
#           regression_trees.RData from 05_Regression_Trees.R (optional)
# PURPOSE:  Identify which interactions to include in improved GLMs (script 04)
# ==============================================================================

source("00_utils.R")
load("benchmark_glms.RData")

# ==============================================================================
# 1. TREE-GUIDED INTERACTION DISCOVERY
# ==============================================================================
# Regression trees naturally capture interactions via their branching structure.
# Variables that split together in the tree are strong interaction candidates.

if (file.exists("regression_trees.RData")) {
  load("regression_trees.RData")
  
  # Variable importance — most influential predictors
  print(tree1$variable.importance)
  
  # Print tree structure — look for which variables split within each branch
  print(tree1)
}

# ==============================================================================
# 2. SYSTEMATIC PAIRWISE INTERACTION SEARCH
# ==============================================================================
# Test every pair of variables as an interaction added to GLM1 (the benchmark).
# Rank by AIC improvement and ANOVA p-value.

base_formula <- ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + 
  VehBrand + VehGas + DensityGLM + Region + AreaGLM

base_model <- glm(base_formula, family = poisson(), data = learn, offset = log(Exposure))
base_aic   <- AIC(base_model)

vars <- c("VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "BonusMalusGLM",
          "VehBrand", "VehGas", "DensityGLM", "Region", "AreaGLM")

interaction_results <- data.frame(
  Interaction = character(), AIC = numeric(), 
  AIC_drop = numeric(), pvalue = numeric(), n_extra_params = integer(),
  stringsAsFactors = FALSE
)

for (i in 1:(length(vars) - 1)) {
  for (j in (i + 1):length(vars)) {
    int_term <- paste(vars[i], vars[j], sep = ":")
    new_formula <- as.formula(paste("ClaimNb ~", paste(vars, collapse = " + "), "+", int_term))
    
    fit <- tryCatch(
      glm(new_formula, family = poisson(), data = learn, offset = log(Exposure)),
      error = function(e) NULL
    )
    
    if (!is.null(fit)) {
      aov  <- anova(base_model, fit, test = "Chisq")
      pval <- aov$`Pr(>Chi)`[2]
      
      interaction_results <- rbind(interaction_results, data.frame(
        Interaction    = int_term,
        AIC            = round(AIC(fit)),
        AIC_drop       = round(base_aic - AIC(fit)),
        pvalue         = signif(pval, 3),
        n_extra_params = length(coef(fit)) - length(coef(base_model)),
        stringsAsFactors = FALSE
      ))
    }
  }
}

# Rank by AIC improvement (biggest drop = most useful)
interaction_results <- interaction_results[order(-interaction_results$AIC_drop), ]

# All pairwise results
print(interaction_results, row.names = FALSE)

# ==============================================================================
# 3. OUT-OF-SAMPLE VALIDATION OF TOP CANDIDATES
# ==============================================================================
# AIC alone can mislead — check out-of-sample deviance for the top interactions.

n_top <- min(10, nrow(interaction_results))
top_interactions <- head(interaction_results, n_top)

oos_results <- data.frame(
  Interaction = character(), InSample = numeric(), OutOfSample = numeric(),
  stringsAsFactors = FALSE
)

for (k in 1:nrow(top_interactions)) {
  int_term <- top_interactions$Interaction[k]
  new_formula <- as.formula(paste("ClaimNb ~", paste(vars, collapse = " + "), "+", int_term))
  
  fit <- glm(new_formula, family = poisson(), data = learn, offset = log(Exposure))
  res <- evaluate_model(fit, int_term, learn, test)
  
  oos_results <- rbind(oos_results, data.frame(
    Interaction = int_term,
    InSample    = res$InSample,
    OutOfSample = res$OutOfSample,
    stringsAsFactors = FALSE
  ))
}

# Add baseline GLM1 for reference
glm1_res <- evaluate_model(glm1, "GLM1 (no interaction)", learn, test)
oos_results <- rbind(
  data.frame(Interaction = "GLM1 (no interaction)", 
             InSample = glm1_res$InSample, OutOfSample = glm1_res$OutOfSample,
             stringsAsFactors = FALSE),
  oos_results
)

# Ranked by out-of-sample deviance
oos_results <- oos_results[order(oos_results$OutOfSample), ]
print(oos_results, row.names = FALSE)

# ==============================================================================
# 4. FORWARD COMBINATION — build up from best single interaction
# ==============================================================================
# Test whether combining the top interactions is better than each alone.
# Add them one at a time and check that each addition still helps.

# Take the top 5 interactions that improved out-of-sample
# (exclude the baseline row)
top_oos <- oos_results[oos_results$Interaction != "GLM1 (no interaction)", ]
top_oos <- head(top_oos, 5)

# Build up incrementally
current_ints <- c()
forward_results <- data.frame(
  Step = character(), Interactions = character(),
  AIC = numeric(), OutOfSample = numeric(), pvalue = character(),
  stringsAsFactors = FALSE
)

prev_model <- base_model

for (k in 1:nrow(top_oos)) {
  candidate <- top_oos$Interaction[k]
  current_ints <- c(current_ints, candidate)
  
  new_formula <- as.formula(paste(
    "ClaimNb ~", paste(vars, collapse = " + "), "+",
    paste(current_ints, collapse = " + ")
  ))
  
  new_model <- tryCatch(
    glm(new_formula, family = poisson(), data = learn, offset = log(Exposure)),
    error = function(e) NULL
  )
  
  if (!is.null(new_model)) {
    aov  <- anova(prev_model, new_model, test = "Chisq")
    pval <- signif(aov$`Pr(>Chi)`[2], 3)
    
    res <- evaluate_model(new_model, paste0("Step ", k), learn, test)
    
    forward_results <- rbind(forward_results, data.frame(
      Step         = paste0("Step ", k),
      Interactions = paste(current_ints, collapse = " + "),
      AIC          = round(AIC(new_model)),
      OutOfSample  = res$OutOfSample,
      pvalue       = as.character(pval),
      stringsAsFactors = FALSE
    ))
    
    prev_model <- new_model
  }
}

# Shows the incremental value of each added interaction
print(forward_results, row.names = FALSE)

# ==============================================================================
# SAVE
# ==============================================================================

save(interaction_results, oos_results, forward_results,
     file = "interaction_search.RData")

write.csv(interaction_results, "interaction_search_all_pairs.csv", row.names = FALSE)
write.csv(forward_results, "interaction_forward_selection.csv", row.names = FALSE)
