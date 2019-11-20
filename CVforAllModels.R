library(caret)
library(pROC)
library(randomForest)
library(MASS)
library(klaR)

models <- c('logistic', 'rf', 'qda', 'nb', 'knn', 'svm')
features <- c('NDAI','logSD','CORR','DF','CF','BF','AF','AN')

accuracy_loss <- function(pred, actual){
  return(mean(pred == actual))
}

precision_loss <- function(pred, actual){
  return(sum(pred == actual & actual == 1)/sum(pred == 1))
}

recall_loss <- function(pred, actual){
  return(sum(pred == actual & actual == 1)/sum(actual == 1))
}

f1_loss <- function(pred, actual){
  p <- precision_loss(pred, actual)
  r <- recall_loss(pred, actual)
  return((2* p * r)/(p + r))
}

auc_loss <- function(pred, actual){
  #tpr <- recall_loss(pred, actual)
  #fpr <- sum(actual == 1 & pred != actual)/sum(actual == 0)
  roc <- roc(actual, pred[, "1"])
  plot(roc)
  return(auc(roc))
}

CVBinaryClassifer <- function(classifier, features, labels, K, data, loss){
  folds <- createFolds(data[,labels], k = K)
  loss_vec <- c()
  formula <- paste(paste(labels, "~"), paste(features, collapse = "+"))
  if (classifier == "logistic"){
    formula <- paste(paste(labels, "~"), paste(features, collapse = "+"))
    for ( f in 1:length(folds) ){
      fit <- glm(formula, family = 'binomial', data = data[-folds[[f]], ])
      pred_prob <- predict(fit, data[folds[[f]], features])
      pred <- ifelse(pred_prob > 0.5, 1, -1)
      loss_vec[f] <- loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else if (classifier == "rf"){
    for ( f in 1:length(folds) ){
      model <- train(as.formula(formula),
                     data = data[-folds[[f]], ],
                     method = 'rf')
      pred <- predict(model, data[folds[[f]], features])
      pred_prod <- predict(model, data[folds[[f]], features], type = "prob")
      loss_vec[f] = loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else if (classifier == "qda"){
    for ( f in 1:length(folds) ){
      model <- train(as.formula(formula),
                     data = data[-folds[[f]], ],
                     method = 'qda')
      pred <- predict(model, data[folds[[f]], features])
      pred_prod <- predict(model, data[folds[[f]], features], type = "prob")
      loss_vec[f] = loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else if (classifier == "nb"){
    for ( f in 1:length(folds) ){
      model <- train(as.formula(formula),
                     data = data[-folds[[f]], ],
                     method = 'nb')
      pred <- predict(model, data[folds[[f]], features])
      pred_prod <- predict(model, data[folds[[f]], features], type = "prob")
      loss_vec[f] = loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else if (classifier == "knn"){
    for ( f in 1:length(folds) ){
      model <- train(as.formula(formula),
                     data = data[-folds[[f]], ],
                     method = 'knn')
      pred <- predict(model, data[folds[[f]], features])
      pred_prod <- predict(model, data[folds[[f]], features], type = "prob")
      loss_vec[f] = loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else if (classifier == "svm"){
    for ( f in 1:length(folds) ){
      model <- train(as.formula(formula),
                     data = data[-folds[[f]], ],
                     method = 'svmLinear2')
      pred <- predict(model, data[folds[[f]], features])
      pred_prod <- predict(model, data[folds[[f]], features], type = "prob")
      loss_vec[f] = loss(pred, data[folds[[f]], labels])
      print(paste("CV score for Fold", f, "is", loss_vec[f]))
    }
  }
  else{
    print("Not a supported classifier")
  }
}
