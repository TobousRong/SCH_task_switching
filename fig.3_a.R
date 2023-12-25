# 读取数据
mydata <- read.csv("D:/taskswitch100/P_values.csv", header = TRUE)

# 创建一个ggplot2图
p <- ggplot(data = mydata, aes(x = Index, y = P_Value)) +
  geom_line(color = "black", size = 0.8) +
  xlab("Modes") +
  ylab("P_value") +
  
  # 设置y轴刻度的字体方向为横向
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white"),
        panel.grid = element_blank(),
        axis.text.x = element_text(size = 20,color = "black"),  # 设置x轴文本字体大小
        axis.text.y = element_text(size = 20,color = "black"),  # 设置y轴文本字体大小
        axis.title.x = element_text(size = 22,color = "black"),  # 设置x轴标题字体大小
        axis.title.y = element_text(size = 22,color = "black"))  # 设置y轴标题字体大小

# 创建筛选出P_Value小于0.05的数据点的子数据集
significant_points <- mydata[mydata$P_Value < 0.05, ]

# 绘制P_Value小于0.05的数据点，并突出显示
p <- p + geom_point(data = significant_points, aes(x = Index, y = P_Value), color = "red", shape = 19, size = 1.5)

# 添加y=0.05的虚线
p <- p + geom_hline(yintercept = 0.05, linetype = "dashed", color = "#999999", size = 1)
p
# 使用ggsave保存为TIFF文件
ggsave("Modes.tiff", p, path = "D:/taskswitch100",width = 8, height = 6, units = "in", dpi = 300)


