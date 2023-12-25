# install.packages("ggplot2")
library(ggplot2)

df <- data.frame(group = c("Multimodel", "Singlemodel"),
                 count = c(0.781617647,0.6875))

Hse <- ggplot(df, aes(x = group, y = count, fill = group)) +
  geom_bar(stat = "identity", width = 0.5, position = position_dodge2(padding = 0.3)) +
  scale_fill_manual(values = c("#87CEFA", "#4682B4")) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white"), 
    panel.grid = element_blank(),  # 隐藏网格线
    axis.line = element_line(color = "black", size = 1),  # 设置坐标轴的颜色为黑色
    axis.text = element_text(size = 15, color = "black"),  # 设置刻度标签的字体大小为12
    axis.title = element_text(size = 16)  # 设置轴标题的字体大小为12
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0.00, 1.00, 0.50)) +
  xlab("") +  # 设置 x 轴标签为空字符
  ylab("Accuracy")+  # 设置 y 轴标签为 "Count" 
theme(legend.position = "none")  # 移除图例
Hse
ggsave("RT_SRT.tiff", path = "D:/taskswitch100/SCHZ/RT")
