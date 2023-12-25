library(ggplot2)
library(ggdendro)
library(ggridges)
library(reshape2)
library(sysfonts)
library(showtextdb)
library(showtext)
library(gridExtra)
library(cowplot)
library(ggbreak)

mydata <- read.csv("D:/taskswitch100/SCH_output.csv",1)##???ݵ???

cor.test(mydata$ts_costlong[1:105],mydata$ts_costshort[1:105])

model<- aov(Hin~age+sex+group,mydata)
summary(model)

##====================
data<-mydata[1:161,]
model<-lm(reaction_time~sex+age+I(^2)+HB,mydata)
summary(model)

###===========================脑区的HB
mydata <- read.csv("D:/taskswitch100/SCH_HB.csv",1)##???ݵ???
age<-mydata$age
sex<-mydata$sex
group<-mydata$group
HB<-mydata[,10:109]
P<-rep(0,100)
for (i in 1:100){
  model<- aov(HB[,i]~age+sex+group)
  X<-summary(model)[[1]]$`Pr(>F)`
  P[i]=X[3]
}
hist(P)
P<-p.adjust(P,method = "fdr")
which(P<0.05)##得到显著变化的模态编号，只考虑
TukeyHSD(model,"group")

###===========================脑区的HB
mydata <- read.csv("D:/taskswitch100/SCH_HF.csv",1)##???ݵ???
age<-mydata$age
sex<-mydata$sex
group<-mydata$group
HF<-mydata[,10:109]
P<-rep(0,100)
for (i in 1:100){
  model<- aov(HF[,i]~age+sex+group)
  X<-summary(model)[[1]]$`Pr(>F)`
  P[i]=X[3]
}
hist(P)
FDR<-p.adjust(P,method = "fdr")
which(P<0.05)##得到显著变化的模态编号，只考虑
TukeyHSD(model,"group")
