library(ggplot2)
mydata <- read.csv("D:/taskswitch100/regression.csv", header = TRUE)


# linear trend + confidence interval
cor_result <- cor.test(mydata$DMN_HB, mydata$reactiontime)
cor_estimate <- cor_result$estimate
cor_p_value <- cor_result$p.value

p<- ggplot(mydata, aes(x =DMN_HB, y =reactiontime)) +
  geom_point(color = "#999999", size =3) +
  geom_smooth(method = lm, color = "black", size =2, fill = "#999999", se = FALSE) +
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 22),
    axis.title.y = element_text(size = 22),
    axis.line = element_line(size = 1),
    axis.text = element_text(size = 20,color = "black"),
  ) +
scale_x_continuous(expression(H[B]), breaks = seq(-0.006, 0.004, 0.004))+
scale_y_continuous(expression(RT))
  #scale_y_continuous(limits = c(0.4, 1.2), breaks = seq(0.4, 1.2, 0.4)) 
p
ggsave("DMN_HB_RT.tiff", p, width = 6, height =5, dpi = 300, device = "tiff",path = "D:/taskswitch100/")

