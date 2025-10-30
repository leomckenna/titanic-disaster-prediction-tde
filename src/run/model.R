# src/run/model.R
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(caret)
})

BANNER <- paste(rep("=", 72), collapse = "")

log_msg <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
  flush.console()
}

get_arg <- function(flag = "--data_dir", default = "src/data") {
  args <- commandArgs(trailingOnly = TRUE)
  hit  <- which(args == flag)
  if (length(hit) == 1 && length(args) >= hit + 1) return(args[hit + 1])
  default
}

safe_read_csv <- function(path) {
  log_msg(paste0("Loading file: ", path))
  df <- readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
  log_msg(paste0("Loaded shape: (", nrow(df), ", ", ncol(df), ")"))
  df
}

add_adjust_features <- function(df, is_train = TRUE) {
  log_msg("15) ADD/ADJUST: Creating features (FamilySize, IsAlone, Title).")

  # Ensure SibSp/Parch exist (placeholders if missing)
  if (!"SibSp" %in% names(df)) {
    df$SibSp <- 0L
    log_msg("Added missing column 'SibSp' = 0 (placeholder).")
  }
  if (!"Parch" %in% names(df)) {
    df$Parch <- 0L
    log_msg("Added missing column 'Parch' = 0 (placeholder).")
  }

  # Family features
  df <- df %>%
    mutate(
      FamilySize = coalesce(SibSp, 0) + coalesce(Parch, 0) + 1,
      IsAlone    = as.integer(FamilySize == 1)
    )

  # Title from Name
  if ("Name" %in% names(df)) {
    extracted <- str_match(coalesce(df$Name, ""), ",\\s*([^\\.]+)\\.")
    df$Title  <- ifelse(is.na(extracted[, 2]), "Unknown", str_trim(extracted[, 2]))
  } else {
    df$Title <- "Unknown"
    log_msg("Column 'Name' not found; setting Title='Unknown'.")
  }

  # Candidate features
  candidate_features <- c(
    # numeric
    "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
    # categorical
    "Sex", "Embarked", "Title"
  )

  # Ensure all candidate columns exist
  for (c in candidate_features) {
    if (!c %in% names(df)) {
      df[[c]] <- NA
      log_msg(paste0("Added missing column '", c, "' as NA (placeholder)."))
    }
  }

  log_msg(paste0("Using features: ", paste(candidate_features, collapse = ", ")))
  list(df = df, features = candidate_features)
}

prep_medians <- function(df, num_cols) {
  vapply(num_cols, function(col) median(df[[col]], na.rm = TRUE), numeric(1))
}

impute_numeric <- function(df, medians) {
  for (col in names(medians)) {
    if (!col %in% names(df)) next
    df[[col]][is.na(df[[col]])] <- medians[[col]]
  }
  df
}

prep_categoricals <- function(df, cat_cols) {
  for (c in cat_cols) {
    if (!c %in% names(df)) next
    df[[c]] <- as.character(df[[c]])
    df[[c]][is.na(df[[c]]) | df[[c]] == ""] <- "Missing"
    df[[c]] <- as.factor(df[[c]])
  }
  df
}

one_hot_fit <- function(df, features) {
  # Build dummy encoder on training features only
  dv <- caret::dummyVars(~ ., data = df[, features, drop = FALSE], fullRank = FALSE)
  dv
}

one_hot_apply <- function(dv, df) {
  as.data.frame(predict(dv, newdata = df))
}

train_glm <- function(X, y) {
  # y must be factor with levels 0/1 (or numeric 0/1)
  dat <- cbind.data.frame(Survived = y, X)
  # Use glm binomial
  glm(Survived ~ ., data = dat, family = binomial())
}

predict_class <- function(model, X, threshold = 0.5) {
  p <- as.numeric(predict(model, newdata = X, type = "response"))
  as.integer(p >= threshold)
}

accuracy <- function(y_true, y_pred) {
  mean(as.integer(y_true) == as.integer(y_pred))
}

main <- function() {
  data_dir <- get_arg("--data_dir", "src/data")
  train_path <- file.path(data_dir, "train.csv")
  test_path  <- file.path(data_dir, "test.csv")

  if (!file.exists(train_path)) stop(paste("ERROR:", train_path, "not found."))
  if (!file.exists(test_path))  stop(paste("ERROR:", test_path, "not found."))

  train <- safe_read_csv(train_path)
  test  <- safe_read_csv(test_path)

  if (!"Survived" %in% names(train)) stop("ERROR: TRAIN must contain 'Survived' column.")

  # 15) Add/Adjust features
  aa_train <- add_adjust_features(train, is_train = TRUE)
  train    <- aa_train$df
  features <- aa_train$features

  aa_test <- add_adjust_features(test, is_train = FALSE)
  test    <- aa_test$df

  # Split X/y
  X_train <- train[, features, drop = FALSE]
  y_train <- as.integer(train$Survived)

  # Identify numeric vs categorical
  num_cols <- c("Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone")
  cat_cols <- setdiff(features, num_cols)

  # Impute numeric by train medians
  medians <- prep_medians(X_train, num_cols)
  X_train <- impute_numeric(X_train, medians)
  X_test  <- test[, features, drop = FALSE]
  X_test  <- impute_numeric(X_test, medians)

  # Prepare categoricals (NA -> "Missing", as.factor)
  X_train <- prep_categoricals(X_train, cat_cols)
  X_test  <- prep_categoricals(X_test,  cat_cols)

  # One-hot encode (fit on train, apply to both)
  dv <- one_hot_fit(X_train, features)
  Xtr <- one_hot_apply(dv, X_train)
  Xte <- one_hot_apply(dv, X_test)

  # 16) Fit & training accuracy
  log_msg("Fitting Logistic Regression on TRAIN...")
  model <- train_glm(Xtr, y_train)
  yhat_train <- predict_class(model, Xtr, threshold = 0.5)
  acc_train  <- accuracy(y_train, yhat_train)
  log_msg(sprintf("16) ACCURACY (TRAIN): %.4f", acc_train))
  log_msg("TRAIN Confusion Matrix:")
  print(table(Truth = y_train, Pred = yhat_train))

  # 17 & 18) Predict & evaluate on TEST (if labels exist)
  if ("Survived" %in% names(test)) {
    log_msg("17) PREDICT: Generating predictions for TEST set with labels present.")
    y_test <- as.integer(test$Survived)
    yhat_test <- predict_class(model, Xte, threshold = 0.5)
    acc_test  <- accuracy(y_test, yhat_test)
    log_msg(sprintf("18) ACCURACY (TEST): %.4f", acc_test))
    log_msg("TEST Confusion Matrix:")
    print(table(Truth = y_test, Pred = yhat_test))
  } else {
    log_msg("17) PREDICT: TEST has no 'Survived' column. Producing predictions only.")
    yhat_test <- predict_class(model, Xte, threshold = 0.5)
    log_msg(paste0("Sample predictions (first 10): ", paste(head(yhat_test, 10), collapse = ", ")))
    log_msg("18) ACCURACY: Skipped because TEST has no labels.")
  }

  log_msg("DONE.")
}

# Run
main()
