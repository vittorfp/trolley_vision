# Programa que vai definir o limiar entre as duas classes

library(ggplot2)

data <- read.csv2('labels.txt',sep = '\n',header = FALSE)
data <- cbind(data,read.csv2('correlation.txt',sep = '\n',header = FALSE) )

names(data) <- c('labels','correlation')
summary(data)

qplot(data = data, x = correlation,  y = labels, geom = c("count"), col = labels )
