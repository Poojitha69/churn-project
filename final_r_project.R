library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(xgboost)
library(pROC)
library(rpart.plot)
library(gridExtra)
library(xgboost)
library(caret)
library(SHAPforxgboost)
library(cluster)
# Load the dataset
data <- read.csv(file.choose(), sep = ',', header = T)
str(data)
summary(data)

# Set 1 Visualization (customerID, gender, SeniorCitizen, Partner, Dependents)
p1 <- ggplot(data, aes(x = customerID)) + 
  geom_bar(fill = 'skyblue') + 
  labs(title = "Distribution of customerID")

p2 <- ggplot(data, aes(x = gender)) + 
  geom_bar(fill = 'lightgreen') + 
  labs(title = "Distribution of gender")

p3 <- ggplot(data, aes(x = SeniorCitizen)) + 
  geom_bar(fill = 'salmon') + 
  labs(title = "Distribution of SeniorCitizen")

p4 <- ggplot(data, aes(x = Partner)) + 
  geom_bar(fill = 'orange') + 
  labs(title = "Distribution of Partner")

p5 <- ggplot(data, aes(x = Dependents)) + 
  geom_bar(fill = 'purple') + 
  labs(title = "Distribution of Dependents")

# Combine Set 1 plots
grid.arrange(p1, p2, p3, p4, p5, ncol = 3)

# Set 2 Visualization (tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity)
p6 <- ggplot(data, aes(x = tenure)) + 
  geom_histogram(bins = 30, fill = 'skyblue', color = 'black') + 
  labs(title = "Distribution of tenure")

p7 <- ggplot(data, aes(x = PhoneService)) + 
  geom_bar(fill = 'lightgreen') + 
  labs(title = "Distribution of PhoneService")

p8 <- ggplot(data, aes(x = MultipleLines)) + 
  geom_bar(fill = 'salmon') + 
  labs(title = "Distribution of MultipleLines")

p9 <- ggplot(data, aes(x = InternetService)) + 
  geom_bar(fill = 'orange') + 
  labs(title = "Distribution of InternetService")

p10 <- ggplot(data, aes(x = OnlineSecurity)) + 
  geom_bar(fill = 'purple') + 
  labs(title = "Distribution of OnlineSecurity")

# Combine Set 2 plots
grid.arrange(p6, p7, p8, p9, p10, ncol = 3)

# Set 3 Visualization (OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
p11 <- ggplot(data, aes(x = OnlineBackup)) + 
  geom_bar(fill = 'skyblue') + 
  labs(title = "Distribution of OnlineBackup")

p12 <- ggplot(data, aes(x = DeviceProtection)) + 
  geom_bar(fill = 'lightgreen') + 
  labs(title = "Distribution of DeviceProtection")

p13 <- ggplot(data, aes(x = TechSupport)) + 
  geom_bar(fill = 'salmon') + 
  labs(title = "Distribution of TechSupport")




p14 <- ggplot(data, aes(x = StreamingTV)) + 
  geom_bar(fill = 'orange') + 
  labs(title = "Distribution of StreamingTV")

p15 <- ggplot(data, aes(x = StreamingMovies)) + 
  geom_bar(fill = 'purple') + 
  labs(title = "Distribution of StreamingMovies")

# Combine Set 3 plots
grid.arrange(p11, p12, p13, p14, p15, ncol = 3)

# Set 4 Visualization (Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)
p16 <- ggplot(data, aes(x = Contract)) + 
  geom_bar(fill = 'skyblue') + 
  labs(title = "Distribution of Contract")

p17 <- ggplot(data, aes(x = PaperlessBilling)) + 
  geom_bar(fill = 'lightgreen') + 
  labs(title = "Distribution of PaperlessBilling")

p18 <- ggplot(data, aes(x = PaymentMethod)) + 
  geom_bar(fill = 'salmon') + 
  labs(title = "Distribution of PaymentMethod")

p19 <- ggplot(data, aes(x = MonthlyCharges)) + 
  geom_histogram(bins = 30, fill = 'orange', color = 'black') + 
  labs(title = "Distribution of MonthlyCharges")

p20 <- ggplot(data, aes(x = TotalCharges)) + 
  geom_histogram(bins = 30, fill = 'purple', color = 'black') + 
  labs(title = "Distribution of TotalCharges")

# Combine Set 4 plots
grid.arrange(p16, p17, p18, p19, p20, ncol = 3)

# Data Cleaning
# Remove customerID as it's not useful for prediction
print(dim(data))  # Display dimensions of data
head(data)        # Display first few rows to verify

data <- data[, !names(data) %in% "customerID"]

# Convert 'Churn' to a factor
data$Churn <- as.factor(data$Churn)

# Convert other categorical variables to factors
categorical_cols <- sapply(data, is.character)
data[categorical_cols] <- lapply(data[categorical_cols], factor)

# Handling Missing Values
data <- na.omit(data)

# Data Visualization
# Distribution of Churn
ggplot(data, aes(x = Churn)) + 
  geom_bar(fill = 'skyblue') + 
  labs(title = "Churn Distribution")

# Boxplot of MonthlyCharges by Churn
ggplot(data, aes(x = Churn, y = MonthlyCharges)) + 
  geom_boxplot() + 
  labs(title = "Monthly Charges by Churn")

# Correlation Heatmap for numerical features
numeric_features <- data %>% select_if(is.numeric)
correlations <- cor(numeric_features)
heatmap(correlations, main = "Correlation Heatmap", col = colorRampPalette(c("red", "white", "blue"))(20))

# Tenure vs Monthly Charges colored by Churn
ggplot(data, aes(x = tenure, y = MonthlyCharges, color = Churn)) +
  geom_point(alpha = 0.6) + 
  labs(title = "Tenure vs Monthly Charges by Churn")

# Distribution of tenure
ggplot(data, aes(x = tenure, fill = Churn)) + 
  geom_histogram(bins = 30, alpha = 0.6) + 
  labs(title = "Distribution of Tenure by Churn")

# Data Normalization
preProcValues <- preProcess(data %>% select_if(is.numeric), method = c("center", "scale"))
data_norm <- predict(preProcValues, data %>% select_if(is.numeric))
data <- cbind(data %>% select_if(~ !is.numeric(.)), data_norm)

# Encode categorical variables as dummy variables for XGBoost
data_xgb <- dummyVars(" ~ .", data = data) %>% predict(data)
data_xgb <- as.data.frame(data_xgb)
data_xgb$Churn <- as.factor(data$Churn)

# Split Data into Training and Testing Sets
set.seed(123)
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# For XGBoost: Use the dummy-encoded data
train_xgb <- data_xgb[trainIndex, ]
test_xgb <- data_xgb[-trainIndex, ]

#1. SVM Model
svm_model <- svm(Churn ~ ., data = train, kernel = "radial", cost = 1, gamma = 0.1)
svm_pred <- predict(svm_model, test)
svm_confusion <- confusionMatrix(svm_pred, test$Churn)
svm_confusion
svm_accuracy <- svm_confusion$overall['Accuracy']
svm_accuracy
selected_features <- data[, c("tenure", "MonthlyCharges", "Churn")]

# Re-train the SVM model using only the two features for visualization purposes
svm_model_2d <- svm(Churn ~ tenure + MonthlyCharges, data = selected_features, type = "C-classification", kernel = "linear")

# Plot the SVM model, highlighting the support vectors, decision boundary, and data points
plot(svm_model_2d, selected_features, 
     tenure ~ MonthlyCharges,
     main = "SVM Decision Boundary with Support Vectors")

# SVM ROC Curve
svm_roc <- roc(test$Churn, as.numeric(svm_pred))
plot(svm_roc, main = "SVM ROC Curve", col = "red", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(svm_roc), 2)), col = "red", lwd = 2)

#2. Decision Tree Model
tree_model <- rpart(Churn ~ ., data = train, method = "class")
tree_pred <- predict(tree_model, test, type = "class")
tree_confusion <- confusionMatrix(tree_pred, test$Churn)
tree_confusion
tree_accuracy <- tree_confusion$overall['Accuracy']
tree_accuracy
# Plot the decision tree
rpart.plot(tree_model, 
           type = 2,               # Draws the split labels at the nodes
           extra = 104,            # Adds information about samples and class
           under = TRUE,           # Displays variable names under the nodes
           faclen = 0,             # Full names for factors
           cex = 0.8,              # Adjusts the size of text
           main = "Decision Tree Representation")

# Decision Tree ROC Curve
tree_roc <- roc(test$Churn, as.numeric(tree_pred))
plot(tree_roc, main = "Decision Tree ROC Curve", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(tree_roc), 2)), col = "green", lwd = 2)

#3. Random Forest Model
rf_model <- randomForest(Churn ~ ., data = train, ntree = 100)
rf_pred <- predict(rf_model, test)
rf_confusion <- confusionMatrix(rf_pred, test$Churn)
rf_confusion
rf_accuracy <- rf_confusion$overall['Accuracy']
rf_accuracy
# Random Forest ROC Curve
rf_roc <- roc(test$Churn, as.numeric(rf_pred))
plot(rf_roc, main = "Random Forest ROC Curve", col = "blue", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(rf_roc), 2)), col = "blue", lwd = 2)

# Feature Importance Plot 
# Random Forest
rf_importance <- randomForest::importance(rf_model)
rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[,1])
ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_bar(stat = "identity", fill = 'tomato') + 
  coord_flip() + 
  labs(title = "Random Forest Feature Importance", x = "Features", y = "Importance")
varImpPlot(rf_model, 
           main = "Random Forest Feature Importance", 
           col = "blue")

# XGBoost Model
train_matrix <- xgb.DMatrix(data = as.matrix(dplyr::select(train_xgb, -Churn)), label = as.numeric(train$Churn) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(dplyr::select(test_xgb, -Churn)), label = as.numeric(test$Churn) - 1)

params <- list(objective = "binary:logistic", eval_metric = "auc", booster = "gbtree", max_depth = 4, eta = 0.1)
xgb_model <- xgboost(params = params, data = train_matrix, nrounds = 100, verbose = 0)
xgb_pred <- predict(xgb_model, test_matrix) > 0.5
xgb_pred <- factor(ifelse(xgb_pred, "Yes", "No"), levels = levels(test$Churn))
xgb_confusion <- confusionMatrix(xgb_pred, test$Churn)
xgb_confusion
xgb_accuracy <- xgb_confusion$overall['Accuracy']
xgb_accuracy
# XGBoost Feature Importance
xgb_importance <- xgb.importance(model = xgb_model)
xgb_importance_df <- data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain)
ggplot(xgb_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_bar(stat = "identity", fill = 'seagreen') + 
  coord_flip() + 
  labs(title = "XGBoost Feature Importance", x = "Features", y = "Importance")

# Visualize the first tree (tree index 0)
xgb.plot.tree(model = xgb_model, trees = 0, show_node_id = TRUE)


# Model Evaluation
accuracy_df <- data.frame(
  Model = c("SVM", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(svm_accuracy, tree_accuracy, rf_accuracy, xgb_accuracy) * 100  # Multiply by 100 for percentage
)

#  Partial Dependence Plots
# Partial dependence plots show the relationship between individual features and the target variable
# Use your actual feature names in place of "tenure" and "MonthlyCharges"

# Arrange plots in a 1x2 grid for side-by-side comparison
par(mfrow = c(1, 2))

# Partial dependence plot for "tenure"
partialPlot(rf_model, train, tenure, "Yes", main = "Partial Dependence on Tenure")

# Partial dependence plot for "MonthlyCharges"
partialPlot(rf_model, train, MonthlyCharges, "Yes", main = "Partial Dependence on Monthly Charges")

# Reset plot layout to default (1 plot per page)
par(mfrow = c(1, 1))


svm_roc <- roc(test$Churn, as.numeric(svm_pred) - 1)
tree_roc <- roc(test$Churn, as.numeric(tree_pred) - 1)
rf_roc <- roc(test$Churn, as.numeric(rf_pred) - 1)
xgb_roc <- roc(test$Churn, as.numeric(xgb_pred) - 1)

# Plot ROC Curve
ggroc(list(SVM = svm_roc, "Decision Tree" = tree_roc, "Random Forest" = rf_roc, "XGBoost" = xgb_roc)) + 
  labs(title = "ROC Curve Comparison") + 
  theme_minimal()

# Compare Accuracies with a Bar Plot
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) + 
  geom_bar(stat = "identity") + 
  labs(title = "Model Comparison by Accuracy") + 
  scale_y_continuous(labels = scales::percent_format(scale = 1))  # Formatting y-axis as percentage

# Print the Best Model
best_model <- accuracy_df[which.max(accuracy_df$Accuracy), "Model"]
cat("Best model based on accuracy is:", best_model, "\n")

# Print accuracy values
accuracy_df

# Select only numeric columns for clustering, excluding 'Churn'
data_clustering <- data %>% select(-Churn) %>% select_if(is.numeric)

# Normalize data
preProc <- preProcess(data_clustering, method = c("center", "scale"))
data_clustering_normalized <- predict(preProc, data_clustering)

#  Determine optimal number of clusters using the Elbow Method
wss <- sapply(1:10, function(k) {
  kmeans(data_clustering_normalized, centers = k, nstart = 10)$tot.withinss
})

# Plot the Elbow Method result
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters (k)", ylab = "Total within-cluster sum of squares",
     main = "Elbow Method for Optimal k")
# Load necessary library
library(cluster)  # For clustering and visualization
library(dplyr)
# Normalize the data to ensure all features contribute equally
preProcValues <- preProcess(data_clustering, method = c("center", "scale"))
data_norm <- predict(preProcValues, data_clustering)

# Calculate the distance matrix (Euclidean distance)
distance_matrix <- dist(data_norm, method = "euclidean")

# Perform Hierarchical Clustering using the Ward's method
hc <- hclust(distance_matrix, method = "ward.D2")

# Plot the Dendrogram
plot(hc, main = "Dendrogram for Hierarchical Clustering", 
     xlab = "", sub = "", cex = 0.6)

# Cut the dendrogram into a desired number of clusters (e.g., 3)
num_clusters <- 4
clusters <- cutree(hc, k = num_clusters)

# Add cluster assignments to the original data
data$Cluster <- as.factor(clusters)

# Visualize clusters with a scatter plot (example with MonthlyCharges and tenure)
ggplot(data, aes(x = tenure, y = MonthlyCharges, color = Cluster)) + 
  geom_point(alpha = 0.6) +
  labs(title = "Hierarchical Clustering of Customers",
       x = "Tenure", y = "Monthly Charges") +
  theme_minimal()
# Apply K-Means with the chosen number of clusters (e.g., k = 3)
set.seed(123)
k <- 4
kmeans_result <- kmeans(data_clustering_normalized, centers = k, nstart = 25)

# Add the cluster assignment to the original data
data$Cluster <- as.factor(kmeans_result$cluster)

# Visualize the clusters (using two features for a 2D plot)
# Here, we'll use tenure and MonthlyCharges as an example
ggplot(data, aes(x = tenure, y = MonthlyCharges, color = Cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(title = "K-Means Clustering (k = 4)", x = "Tenure", y = "Monthly Charges") +
  theme_minimal()




         