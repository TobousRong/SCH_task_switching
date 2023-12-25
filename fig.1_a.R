library(ggplot2)
library(ggpubr)
library(cowplot)

mydata <- read.csv("D:/taskswitch100/H.csv",header = T)
###=========================================
data<-mydata
Hin <- ggplot(data,aes(group,absHB))+theme_classic()+
  geom_boxplot(color="black",size=1,show.legend = F,width = 0.2,position = position_dodge(width = 0.3))+
  geom_jitter(aes(color=group),size=2.5,position = position_jitter(0.2),show.legend = F,alpha = 0.7)+
  scale_color_manual(values=c("#555555","#999999"))+
  theme(axis.line = element_line(size = 0.7),
        axis.title.y=element_text(color="black",size=16),
        axis.title.x=element_blank(),
        axis.text = element_text(color="black",size=18),
        legend.title=element_blank(),
        legend.position = "right",
        legend.text = element_text(size = 18))+
  scale_y_continuous(expression(RT))
Hin

#Δ,breaks = seq(0.4,1.4,0.4)
ggsave("absHB.tiff",path = "D:/taskswitch100/Figure")