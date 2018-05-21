library(ggplot2)

setwd("~/Desktop/Trolley_vision/src/data_analisis/")
data <- read.csv2('../dataset_manager/correntes.txt',sep = ',',header = FALSE,dec = '.')
labels <- read.csv2('../dataset_manager/corrente_labels.txt',sep = '\n',header = FALSE,dec = '.')
frame <- c(1:length(data[,1]))
data <- cbind(data, frame,labels)
names(data) <- c('elo','malha','arrastador','trolley','frame','labels')


head(data)
summary(data)

# Plota a matriz de scatter
plot(data)

qplot(data = data, x = frame, y = elo, geom = c('line') )
qplot(data = data, x = frame, y = malha, geom = c('line') )
qplot(data = data, x = frame, y = arrastador, geom = c('line') )

qplot(data = data, x = malha, geom = c('histogram'), binwidth = 10)

qplot(elo, data = data , fill = labels , facets = labels ~ ., bins = 100)
qplot(malha, data = data , fill = labels , facets = labels ~ ., bins = 100)
qplot(arrastador, data = data , fill = labels , facets = labels ~ ., bins = 100)
