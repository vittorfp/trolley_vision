# Programa que vai definir o limiar entre as duas classes
library(ggplot2)

setwd("~/Desktop/Trolley_vision/src/data_analisis/")
data <- read.csv2('../dataset_manager/labels.txt',sep = '\n',header = FALSE)
data <- cbind(data,read.csv2('../dataset_manager/correlation.txt',sep = '\n',header = FALSE,dec = '.') )

names(data) <- c('labels','correlation')
data$correlation <- as.numeric(data$correlation)
data$labels <- as.factor(data$labels)
summary(data)


# Plot da evolução temporal com o limiar
plot(data$correlation, type = 'line', ylim = range(-1000000:13000000))
abline(a = 6500000, b = 0, col = 'red',lty = 12)


# Plot das distribuições
qplot(correlation, data = data , fill = labels , facets = labels ~ ., bins = 100)
qplot(correlation, data = subset(data, correlation > 0.7e7) , fill = labels , facets = labels ~ ., bins = 100)

# Regressão logistica

tr <- sample(1:length(data$labels), as.integer(length(data[,1]) *0.9)  )
train <- data[tr,]
test <- data[-tr,]

model <- glm(labels ~ correlation,family=binomial(link='logit'), data = train)
summary(model)

# Avalia o modelo
library(ROCR)
p <- predict(model, newdata = test, type="response")
pr <- prediction( p, test$labels )
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc