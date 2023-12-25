library(ggplot2)
mydata <- read.csv("D:/taskswitch100/RT_H.csv", header = TRUE)

p <- ggplot(mydata, aes(x = HB, y =reaction_time)) +
  geom_point(data = mydata[1:111, ], aes(x = HB, y = reaction_time), color = "#555555", size = 2.5) +
  geom_point(data = mydata[112:161, ], aes(x = HB, y = reaction_time), color =  "#999999", size = 2.5) +
  geom_smooth(data = mydata, method = "lm", formula = y ~ poly(x, 2), color = "black", size = 1.5, fill = "#999999", se = FALSE) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray", size = 1) +  # 添加虚线
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 22),
    axis.title.y = element_text(size = 22),
    axis.line = element_line(size = 1),
    axis.text = element_text(size = 20, color = "black"),
    plot.margin = margin(20, 20, 15, 15) # 根据需要调整边距
  ) +
  scale_x_continuous(expression(H[B])) +
  scale_y_continuous(expression(RT))
p

#,breaks = seq(0.000,0.005,0.002)
ggsave("RT_HB.tiff", p, width = 6, height =5, dpi = 300, device = "tiff",path = "D:/taskswitch100/")