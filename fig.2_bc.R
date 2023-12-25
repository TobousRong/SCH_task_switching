mydata <- read.csv("D:/taskswitch100/RT_H.csv", header = TRUE)

# linear trend + confidence interval
cor_result <- cor.test(mydata$Hin, mydata$reaction_time)
cor_estimate <- cor_result$estimate
cor_p_value <- cor_result$p.value
p <- ggplot() +
  geom_point(data = mydata[1:111, ], aes(x = Hin, y = reaction_time), color = "#555555", size = 2.5) +
  geom_point(data = mydata[112:161, ], aes(x = Hin, y = reaction_time), color =  "#999999", size = 2.5) +
  geom_smooth(data = mydata, method = lm, aes(x = Hin, y = reaction_time), color = "black", size = 2, fill = "#999999", se = FALSE) +
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 22),
    axis.title.y = element_text(size = 22),
    axis.line = element_line(size = 1),
    axis.text = element_text(size = 20, color = "black"),
    #plot.margin = margin(20, 20, 15, 15)
  ) +
  scale_x_continuous(expression(H[In])) +
  scale_y_continuous(expression(RT))
p
ggsave("RT_Hin.tiff", p,  width = 6, height =5, dpi = 300,device = "tiff",path = "D:/taskswitch100/")