library(xgboost)
library(caret)
library(survival)
library(ParBayesianOptimization)
library(timeROC)
library(SHAPforxgboost)


## Prepare design matrix
features <- setdiff(colnames(df), c("time", "dem"))
X_full <- model.matrix(~ . - 1, data = df[, ..features])
time_full <- df$time
event_full <- df$dem

dem_freq_full <- mean(event_full == 1)
weights_full <- ifelse(event_full == 1, 1 / dem_freq_full, 1 / (1 - dem_freq_full))


## Nested CV
K_outer <- 5
K_inner <- 3
eval_times <- 1:7
SEED <- 999

set.seed(SEED)
outer_folds <- createFolds(event_full, k = K_outer, list = TRUE, returnTrain = FALSE)

run_outer_fold <- function(k) {
  set.seed(SEED)
  
  test_idx <- outer_folds[[k]]
  train_idx <- setdiff(seq_len(nrow(X_full)), test_idx)
  
  X_outer_train <- X_full[train_idx, , drop = FALSE]
  time_outer_train <- time_full[train_idx]
  event_outer_train <- event_full[train_idx]
  weights_outer_train <- weights_full[train_idx]
  
  X_outer_test <- X_full[test_idx, , drop = FALSE]
  time_outer_test <- time_full[test_idx]
  event_outer_test <- event_full[test_idx]
  
  set.seed(SEED)
  inner_folds <- createFolds(event_outer_train, k = K_inner, list = TRUE, returnTrain = FALSE)
  
  scoringFunction <- function(max_depth, eta, subsample, colsample_bytree,
                              gamma, lambda, alpha, min_child_weight) {
    params <- list(
      booster = "gbtree",
      objective = "survival:aft",
      eval_metric = "aft-nloglik",
      aft_loss_distribution = "normal",
      aft_loss_distribution_scale = 1.0,
      max_depth = as.integer(max_depth),
      eta = eta,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      lambda = lambda,
      alpha = alpha,
      min_child_weight = as.integer(min_child_weight),
      nthread = 1
    )
    
    cv_scores <- numeric(length(inner_folds))
    
    for (i in seq_along(inner_folds)) {
      val_idx <- inner_folds[[i]]
      
      if (sum(event_outer_train[val_idx] == 1) == 0) {
        cv_scores[i] <- NA
        next
      }
      
      dtrain <- xgb.DMatrix(
        data = X_outer_train[-val_idx, , drop = FALSE],
        label_lower_bound = time_outer_train[-val_idx],
        label_upper_bound = ifelse(event_outer_train[-val_idx] == 1,
                                   time_outer_train[-val_idx], Inf),
        weight = weights_outer_train[-val_idx]
      )
      dval <- xgb.DMatrix(
        data = X_outer_train[val_idx, , drop = FALSE],
        label_lower_bound = time_outer_train[val_idx],
        label_upper_bound = ifelse(event_outer_train[val_idx] == 1,
                                   time_outer_train[val_idx], Inf)
      )
      
      model <- tryCatch(
        xgb.train(
          params = params,
          data = dtrain,
          nrounds = 300,
          evals = list(val = dval),
          callbacks = list(xgb.cb.evaluation.log(),
                           xgb.cb.early.stop(20)),
          verbose = 0
        ),
        error = function(e) NULL
      )
      
      if (is.null(model)) {
        cv_scores[i] <- NA
        next
      }
      
      log_df <- attributes(model)$evaluation_log
      if (is.null(log_df) || nrow(log_df) == 0) {
        cv_scores[i] <- NA
        next
      }
      
      val_col <- grep("^val_", names(log_df), value = TRUE)[1]
      if (length(val_col) == 0 || is.na(val_col)) {
        cv_scores[i] <- NA
        next
      }
      
      best_score <- min(as.numeric(log_df[[val_col]]), na.rm = TRUE)
      cv_scores[i] <- if (is.finite(best_score)) best_score else NA
    }
    
    mean_cv <- mean(cv_scores, na.rm = TRUE)
    if (!is.finite(mean_cv)) mean_cv <- 1e6
    
    list(Score = -mean_cv)
  }
  
  bounds <- list(
    max_depth = c(3L, 8L),
    eta = c(0.01, 0.03),
    subsample = c(0.7, 0.9),
    colsample_bytree = c(0.7, 1.0),
    gamma = c(0, 2),
    lambda = c(0, 2),
    alpha = c(0, 2),
    min_child_weight = c(5L, 8L)
  )
  
  optObj <- bayesOpt(
    FUN = scoringFunction,
    bounds = bounds,
    initPoints = 10,
    iters.n = 10,
    parallel = FALSE,
    verbose = 0
  )
  
  best_params <- getBestPars(optObj)
  
  dtrain_final <- xgb.DMatrix(
    data = X_outer_train,
    label_lower_bound = time_outer_train,
    label_upper_bound = ifelse(event_outer_train == 1, time_outer_train, Inf),
    weight = weights_outer_train
  )
  
  final_model <- xgb.train(
    params = c(best_params, list(
      booster = "gbtree",
      objective = "survival:aft",
      eval_metric = "aft-nloglik",
      aft_loss_distribution = "normal",
      aft_loss_distribution_scale = 1.0
    )),
    data = dtrain_final,
    nrounds = 300,
    verbose = 0
  )
  
  dtest_final <- xgb.DMatrix(X_outer_test)
  pred <- predict(final_model, dtest_final)
  
  cidx <- concordance(Surv(time_outer_test, event_outer_test) ~ pred)$concordance
  
  roc <- tryCatch(
    timeROC(
      T = time_outer_test,
      delta = event_outer_test,
      marker = -pred,
      cause = 1,
      times = eval_times,
      iid = TRUE
    ),
    error = function(e) NULL
  )
  
  if (!is.null(roc)) {
    auc_times <- roc$AUC
    integrated_auc <- mean(roc$AUC, na.rm = TRUE)
  } else {
    auc_times <- rep(NA_real_, length(eval_times))
    integrated_auc <- NA_real_
  }
  
  shap_test <- predict(final_model, dtest_final, predcontrib = TRUE)
  
  list(
    cindex = cidx,
    best_params = best_params,
    auc_times = auc_times,
    integrated_auc = integrated_auc,
    shap_test = shap_test,
    X_test = X_outer_test,
    final_model = final_model,
    pred = pred,
    time_test = time_outer_test,
    event_test = event_outer_test
  )
}


## Matrices
results_nested_list <- lapply(seq_along(outer_folds), run_outer_fold)

summarize_nested_cv <- function(results_nested_list, eval_times) {
  K <- length(results_nested_list)
  
  cidx_outer <- sapply(results_nested_list, `[[`, "cindex")
  mean_cidx <- mean(cidx_outer, na.rm = TRUE)
  sd_cidx <- sd(cidx_outer, na.rm = TRUE)
  ci_cidx <- mean_cidx + c(-1, 1) * 1.96 * sd_cidx / sqrt(K)
  
  auc_matrix <- do.call(rbind, lapply(results_nested_list, `[[`, "auc_times"))
  mean_auc <- colMeans(auc_matrix, na.rm = TRUE)
  sd_auc <- apply(auc_matrix, 2, sd, na.rm = TRUE)
  ci_width <- 1.96 * sd_auc / sqrt(K)
  ci_auc_lower <- mean_auc - ci_width
  ci_auc_upper <- mean_auc + ci_width
  
  int_auc_outer <- sapply(results_nested_list, `[[`, "integrated_auc")
  mean_int_auc <- mean(int_auc_outer, na.rm = TRUE)
  sd_int_auc <- sd(int_auc_outer, na.rm = TRUE)
  ci_int_auc <- mean_int_auc + c(-1, 1) * 1.96 * sd_int_auc / sqrt(K)
  
  list(
    cindex = list(mean = mean_cidx, ci = ci_cidx),
    auc_times = list(mean = mean_auc, lower = ci_auc_lower, upper = ci_auc_upper),
    integrated_auc = list(mean = mean_int_auc, ci = ci_int_auc)
  )
}

summary_out <- summarize_nested_cv(results_nested_list, eval_times)


## SHAP plot
all_shap_std <- do.call(rbind, lapply(results_nested_list, function(x) {
  s <- x$shap_test
  s <- s[, -ncol(s), drop = FALSE]
  s <- s / mean(abs(s))
  as.data.frame(s)
}))

all_X_test <- do.call(rbind, lapply(results_nested_list, function(x) {
  as.data.frame(x$X_test)
}))

shap_plot <- shap.plot.summary.wrap2(
  shap_score = all_shap_std,
  X          = all_X_test,
  top_n      = 15
)


