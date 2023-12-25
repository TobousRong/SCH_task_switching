rm(list=ls())
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(RColorBrewer)
library(cowplot)
library(grid)
library(lmtest)



data1 <- read.csv("D:/taskswitch100/regression.csv", header = T)

data<-data1[,11:34]
score <-data1[,3:9]

output <- as.data.frame(matrix(NA, ncol(score),ncol(data)))

for (s in seq_along(score)) {
  for (i in seq_along(data)){
    data2 <- data.frame(data[,i],score[,s],age = data1[, 3],sex = data1[, 4])
    colnames(data2)[1] <- "x"
    colnames(data2)[2] <- "y"
    my_formular <- lm(y ~ sex+age+x ,data2)
    my_formular2 <- lm(y ~sex+age+I(x^2)+x,data2)
    results <- lrtest(my_formular, my_formular2)
    output[[s,i]] <- results["2","Pr(>Chisq)"]
  }}
colnames(output) <- colnames(data)
rownames(output) <- colnames(score)
write.csv(output,file = "D:/taskswitch100/lrtest.csv")