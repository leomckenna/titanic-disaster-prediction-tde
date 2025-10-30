# src/R_run/model.R
slog <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
  flush.console()
}

args <- commandArgs(trailingOnly = TRUE)
data_dir <- "src/data"
if (length(args) >= 2 && args[1] == "--data_dir") data_dir <- args[2]

train_path  <- file.path(data_dir, "train.csv")
test_path   <- file.path(data_dir, "test.csv")

# ---- Load ----
slog(paste("Loading file:", train_path))
suppressWarnings({
  train <- tryCatch(data.table::fread(train_path), error = function(e) read.csv(train_path))
})
slog(paste("Loaded shape:", sprintf("(%d, %d)", nrow(train), ncol(train))))

slog(paste("Loading file:", test_path))
suppressWarnings({
  test  <- tryCatch(data.table::fread(test_path),  error = function(e) read.csv(test_path))
})
slog(paste("Loaded shape:", sprintf("(%d, %d)", nrow(test), ncol(test))))

if (!"Survived" %in% names(train)) stop("ERROR: TRAIN must contain 'Survived' column.")

# ---- 15) Feature engineering ----
add_feats <- function(df, which="DATA") {
  slog(sprintf("15) ADD/ADJUST (%s): Creating features (FamilySize, IsAlone, Title).", which))
  if (!"SibSp" %in% names(df)) df$SibSp <- 0L
  if (!"Parch" %in% names(df)) df$Parch <- 0L

  df$FamilySize <- ifelse(is.na(df$SibSp), 0, df$SibSp) +
                   ifelse(is.na(df$Parch), 0, df$Parch) + 1L
  df$IsAlone <- as.integer(df$FamilySize == 1L)

  if ("Name" %in% names(df)) {
    m <- regexec(",\\s*([^\\.]+)\\.", df$Name)
    ext <- regmatches(df$Name, m)
    df$Title <- vapply(ext, function(x) if (length(x) >= 2) trimws(x[2]) else "Unknown", character(1))
  } else df$Title <- "Unknown"

  needed <- c("Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","Sex","Embarked","Title")
  for (c in needed) if (!c %in% names(df)) df[[c]] <- NA

  med_or <- function(x) {
    x <- as.numeric(x)
    if (all(is.na(x))) { x[is.na(x)] <- 0; return(x) }
    m <- stats::median(x, na.rm = TRUE)
    x[is.na(x)] <- m
    x
  }
  if ("Age" %in% names(df))  df$Age  <- med_or(df$Age)
  if ("Fare"%in% names(df))  df$Fare <- med_or(df$Fare)
  for (c in c("Pclass","SibSp","Parch","FamilySize","IsAlone")) if (c %in% names(df)) df[[c]] <- as.numeric(df[[c]])

  # normalize rare titles + align later
  normalize_titles <- function(x) {
    x <- as.character(x)
    x[x %in% c("Mlle","Ms")] <- "Miss"
    x[x == "Mme"] <- "Mrs"
    rare <- c("Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona")
    x[x %in% rare] <- "Rare"
    factor(x)
  }
  df$Title <- normalize_titles(df$Title)

  if ("Embarked" %in% names(df)) {
    mode_emb <- names(sort(table(df$Embarked), decreasing=TRUE))[1]
    df$Embarked[is.na(df$Embarked) | df$Embarked==""] <- mode_emb
  }
  df$Sex      <- factor(df$Sex)
  df$Embarked <- factor(df$Embarked)
  df$Title    <- factor(df$Title)
  df
}

train <- add_feats(train, "TRAIN")
test  <- add_feats(test,  "TEST")

# Align factor levels: make TEST use TRAIN's levels
align_to_train <- function(x, train_levels) {
  x <- as.character(x)
  x[!(x %in% train_levels)] <- train_levels[1]
  factor(x, levels = train_levels)
}
test$Sex      <- align_to_train(test$Sex,      levels(train$Sex))
test$Embarked <- align_to_train(test$Embarked, levels(train$Embarked))
test$Title    <- align_to_train(test$Title,    levels(train$Title))

features <- c("Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","Sex","Embarked","Title")
slog(paste("Using features:", paste(features, collapse=", ")))

# ---- 16) Fit GLM on TRAIN and report accuracy ----
form <- as.formula(paste("Survived ~", paste(features, collapse=" + ")))
slog("Building GLM (logistic regression).")
fit <- glm(form, data=train, family=binomial())

pred_train <- as.integer(predict(fit, newdata=train, type="response") >= 0.5)
acc_train  <- mean(pred_train == train$Survived)
cm_train   <- table(Truth=train$Survived, Pred=pred_train)
slog(sprintf("16) ACCURACY (TRAIN): %.4f", acc_train))
slog("TRAIN Confusion Matrix:")
print(cm_train)

# ---- 17) Predict TEST and save CSV ----
pred_test <- as.integer(predict(fit, newdata=test, type="response") >= 0.5)
if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)
out_path <- "outputs/submission_r.csv"
utils::write.csv(
  data.frame(PassengerId = test$PassengerId, Survived = pred_test),
  out_path, row.names = FALSE
)
slog(paste("17) PREDICT: Wrote predictions to", out_path))

# ---- 18) Per instructor: skip test accuracy ----
slog("18) ACCURACY (TEST): Skipped per instructions (save predictions only).")
slog("DONE.")
