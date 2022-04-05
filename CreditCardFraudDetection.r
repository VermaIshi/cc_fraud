#install.packages("dplyr")
library(dplyr) # for data manipulation
#install.packages("stringr")
library(stringr) # for data manipulation
#install.packages("caret")
library(caret) # for sampling
#install.packages("caTools")
library(caTools) # for train/test split
#install.packages("ggplot2")
library(ggplot2) # for data visualization
#install.packages("corrplot")
library(corrplot) # for correlations
#install.packages("Rtsne")
library(Rtsne) # for tsne plotting
#install.packages("DMwR")
library("DMwR") # for smote implementation
#install.packages("ROSE")
library(ROSE)# for ROSE sampling
#install.packages("rpart")
library(rpart)# for decision tree model
#install.packages("xgboost")
library(xgboost) # for xgboost model
#install.packages("neuralnet")
library(pROC)

#Loading Data
credit_card <-read.csv("C:/Users/hitss/Desktop/creditcard.csv")

#Data Exploration
head(credit_card)
str(credit_card)
summary(credit_card)
#checking missing values
colSums(is.na(credit_card))
# checking class imbalance
table(credit_card$Class)

fig(12, 8)
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = credit_card, aes(x = factor(Class), 
                               y = prop.table(stat(count)), fill = factor(Class),
                               label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme

#Distribution of variable 'Time' by class

fig(14, 8)
credit_card %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y') + common_theme

#Distribution of Amount with Class
fig(14, 8)
ggplot(credit_card, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") + common_theme


#Correlation of anonymised variables and 'Amount'¶
fig(14, 8)
correlations <- cor(credit_card[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

#Remove Time variable
credit_card <- credit_card[,-1]
#Change 'Class' variable to factor
credit_card$Class <- as.factor(credit_card$Class)
levels(credit_card$Class) <- c("Not_Fraud", "Fraud")

#Scale numeric variables

credit_card[,-30] <- scale(credit_card[,-30])

head(credit_card)


#Spliting the data

set.seed(123)
split <- sample.split(credit_card$Class, SplitRatio = 0.7)
train <-  subset(credit_card, split == TRUE)
test <- subset(credit_card, split == FALSE)


# class ratio initially in train
table(train$Class)


# downsampling
set.seed(9560)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)

# upsampling
set.seed(9560)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

# smote
set.seed(9560)
smote_train <- SMOTE(Class ~ ., data  = train)

# rose
set.seed(9560)
rose_train <- ROSE(Class ~ ., data  = train)$data 

table(rose_train$Class)

#Building Models

#Logistic Regression
glm_fit <- glm(Class ~ ., data = up_train, family = 'binomial')
summary(glm_fit)

pred_glm <- predict(glm_fit, newdata = test, type = 'response')

roc.curve(test$Class, pred_glm, plotit = TRUE)

#Decision Tree with SMOTE
set.seed(5627)
smote_fit <- rpart(Class ~ ., data = smote_train)

# AUC on SMOTE data
pred_smote <- predict(smote_fit, newdata = test)
print('Fitting model to smote data')
roc.curve(test$Class, pred_smote[,2], plotit = FALSE)

#XGBoost
# Convert class labels from factor to numeric

labels <- up_train$Class

y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)

set.seed(42)
xgb <- xgboost(data = data.matrix(up_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)

xgb_pred <- predict(xgb, data.matrix(test[,-30]))

roc.curve(test$Class, xgb_pred, plotit = TRUE)


#Neural Network

library(neuralnet)
nn=neuralnet(Class~.,data=smote_train, hidden=3, act.fct = "logistic", linear.output = FALSE)
Predict=neuralnet::compute(nn,test)
roc.curve(test$Class, Predict$net.result[,2], plotit = TRUE)



