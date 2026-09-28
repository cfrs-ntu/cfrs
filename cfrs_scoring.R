library(AutoScore)
library(caret)
library(dplyr)
library(survival)
library(timeROC)
library(survcomp)


## Prepare model matrix
time_points <- 1:7

set.seed(555)
features <- setdiff(colnames(df), c("time", "dem"))

train_index <- createDataPartition(df$dem, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

train_mm <- model.matrix(~ . - 1, data = train_data[, ..features])
train_set <- as.data.frame(train_mm) %>%
  mutate(label_time = train_data$time,
         label_status = train_data$dem)


## AutoScore weighting and fine-tuning
cut_vec_init <- AutoScore_weighting_Survival(
  train_set = train_set,
  validation_set = train_set,
  final_variables = selected_features,
  max_score = 100,
  categorize = "quantile",
  quantiles = c(0, 1/3, 2/3, 1),
  time_point = 7
)

scoring_table_init <- AutoScore_fine_tuning_Survival(
  train_set = train_set,
  validation_set = train_set,
  final_variables = selected_features,
  cut_vec = cut_vec_init,
  max_score = 100,
  time_point = time_points
)


## Manual cut-point adjustment
# Inspect scoring_table_init and define manual overrides for predictors
# Example structure:

#   manual_overrides <- list(
#     variable_name = c(-Inf, lower_cut, upper_cut, Inf),
#     ...
#   )

manual_overrides <- list()

cut_vec_manual <- cut_vec_init
for (v in names(manual_overrides)) {
  if (v %in% names(cut_vec_manual)) cut_vec_manual[[v]] <- manual_overrides[[v]]
}

scoring_table_manual <- AutoScore_fine_tuning_Survival(
  train_set = train_set,
  validation_set = train_set,
  final_variables = selected_features,
  cut_vec = cut_vec_manual,
  max_score = 100,
  time_point = time_points
)


## Build score lookup and prediction functions
build_score_lookup <- function(cut_vec, scoring_vec) {
  lookup <- list()
  for (var in names(cut_vec)) {
    idx <- grepl(paste0("^", var), names(scoring_vec))
    lookup[[var]] <- list(
      cuts = cut_vec[[var]],
      scores = as.numeric(scoring_vec[idx])
    )
  }
  lookup
}

predict_score <- function(data, features, lookup) {
  n <- nrow(data)
  total_score <- numeric(n)
  for (var in features) {
    x <- data[[var]]
    cuts <- lookup[[var]]$cuts
    scores <- lookup[[var]]$scores
    bin <- findInterval(x, cuts, rightmost.closed = TRUE) + 1
    add <- scores[bin]
    add[is.na(add)] <- 0
    total_score <- total_score + add
  }
  data.frame(
    time = data$time,
    status = data$dem,
    scor = total_score
  )
}

compute_cindex <- function(df) {
  survcomp::concordance.index(
    x = df$score,
    surv.time = df$time,
    surv.event = df$status,
    method = "noether"
  )$c.index
}

compute_auc <- function(df) {
  roc <- tryCatch(
    timeROC(
      T = df$time,
      delta = df$status,
      marker = df$score,
      cause = 1,
      times = time_points,
      iid = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(roc)) return(NA_real_)
  mean(roc$AUC, na.rm = TRUE)
}

score_lookup <- build_score_lookup(cut_vec_manual, scoring_table_manual)


## Apparent performance on training set
pred_train <- predict_score(train_data, selected_features, score_lookup)

apparent_cindex <- compute_cindex(pred_train)
apparent_iAUC   <- compute_auc(pred_train)


## Bootstrap optimism correction
B <- 300
set.seed(666)

optim_mat <- matrix(NA_real_, nrow = B, ncol = 2)
colnames(optim_mat) <- c("cindex", "iAUC")

for (b in seq_len(B)) {
  
  idx       <- sample.int(nrow(train_data), replace = TRUE)
  boot_data <- train_data[idx, ]
  
  boot_mm  <- model.matrix(~ . - 1, data = boot_data[, ..features])
  boot_set <- as.data.frame(boot_mm) %>%
    mutate(label_time   = boot_data$time,
           label_status = boot_data$dem)
  
  cut_vec_boot <- tryCatch(
    AutoScore_weighting_Survival(
      train_set       = boot_set,
      validation_set  = boot_set,
      final_variables = selected_features,
      max_score       = 100,
      categorize      = "quantile",
      quantiles       = c(0, 1/3, 2/3, 1),
      time_point      = 7
    ),
    error = function(e) NULL
  )
  if (is.null(cut_vec_boot)) next
  
  for (v in names(manual_overrides)) {
    if (v %in% names(cut_vec_boot)) cut_vec_boot[[v]] <- manual_overrides[[v]]
  }
  
  scoring_boot <- tryCatch(
    AutoScore_fine_tuning_Survival(
      train_set       = boot_set,
      validation_set  = boot_set,
      final_variables = selected_features,
      cut_vec         = cut_vec_boot,
      max_score       = 100,
      time_point      = time_points
    ),
    error = function(e) NULL
  )
  if (is.null(scoring_boot)) next
  
  lookup_boot <- build_score_lookup(cut_vec_boot, scoring_boot)
  
  pred_boot_app  <- predict_score(boot_data,  selected_features, lookup_boot)
  pred_boot_orig <- predict_score(train_data, selected_features, lookup_boot)
  
  cindex_app  <- tryCatch(compute_cindex(pred_boot_app),  error = function(e) NA_real_)
  cindex_orig <- tryCatch(compute_cindex(pred_boot_orig), error = function(e) NA_real_)
  iauc_app    <- tryCatch(compute_auc(pred_boot_app),     error = function(e) NA_real_)
  iauc_orig   <- tryCatch(compute_auc(pred_boot_orig),    error = function(e) NA_real_)
  
  optim_mat[b, ] <- c(
    cindex = cindex_app - cindex_orig,
    iAUC   = iauc_app   - iauc_orig
  )
}


## Held-out test set evaluation
pred_test <- predict_score(test_data, selected_features, score_lookup)

test_cindex <- compute_cindex(pred_test)
test_iAUC   <- compute_auc(pred_test)


## Summary table
mean_opt <- colMeans(optim_mat, na.rm = TRUE)

apparent_vals  <- c(apparent_cindex, apparent_iAUC)
test_vals      <- c(test_cindex, test_iAUC)
corrected_vals <- apparent_vals - mean_opt

ci_mat <- matrix(NA_real_, nrow = 2, ncol = length(apparent_vals))
for (i in seq_along(apparent_vals)) {
  corrected_boot <- apparent_vals[i] - optim_mat[, i]
  ci_mat[, i] <- quantile(corrected_boot, probs = c(0.025, 0.975), na.rm = TRUE)
}

results <- data.frame(
  Metric          = c("C-index", "iAUC"),
  Train_Apparent  = round(apparent_vals, 4),
  Optimism        = round(mean_opt, 4),
  Train_Corrected = round(corrected_vals, 4),
  Corrected_Lower = round(ci_mat[1, ], 4),
  Corrected_Upper = round(ci_mat[2, ], 4),
  Test_Set        = round(test_vals, 4)
)

