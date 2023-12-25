library(ggplot2)
# 读取数据
df <- read.csv("D:/taskswitch100/SCHZ/Predication/SANS_sys.csv", header = TRUE)
# 创建绘图
p <- ggplot(df, aes(x = Name, y = Weight)) +
  geom_segment(aes(x = Name, xend = Name, y = 0, yend = Weight),
               color = "gray", lwd = 1.5) +
  geom_point(size = 4, pch = 21, bg =  "#999999",colour="black") +
  coord_flip() +
  theme_minimal()
# 添加边框
p <-p + theme(axis.text.x = element_text(size = 20,color = "black"),  # 设置x轴文本字体大小
          axis.text.y = element_text(size = 20,color = "black"),  # 设置y轴文本字体大小
          axis.title.x = element_text(size = 22,color = "black"),  # 设置x轴标题字体大小
          axis.title.y = element_text(size = 22,color = "black"),  # 设置y轴标题字体大小
  panel.border = element_rect(color = "gray", size = 1, fill = NA)
)
p
ggsave("SANS_sys.tiff", p, path = "D:/taskswitch100/SCHZ/Predication",width =6, height = 5, units = "in", dpi = 300)
