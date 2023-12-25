library(ggplot2)
library(reshape2)

file_path <- "D:/taskswitch100/SCHZ/RT/noRT_performance.csv"

data <- read.csv(file_path)
melted_data <- melt(data, id.vars="Model")


# Create the plot
p <- ggplot(melted_data, aes(x=Model, y=value, fill=variable)) +
  geom_bar(stat="identity", position=position_dodge(width=0.75), width=0.75,alpha = 0.9) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by=0.2)) +
  theme_minimal(base_size = 17) +
  theme(
    axis.text.x = element_text(angle=45, hjust=.8, vjust=.8),
    panel.grid.major = element_line(colour = "grey90"),
    panel.grid.minor = element_blank(),
    legend.position = c(1, 1),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.title = element_blank()
  ) +
  labs(
    x = NULL,
    y = "Score"
  ) +
  scale_fill_manual(values = c("Accuracy" = "#87CEFA", "AUC" = "#4682B4"))

p

dir_path <- dirname(file_path)
img_path <- file.path(dir_path, "noRTmodel_performance_plot.png")
ggsave(filename = img_path, plot = p, width = 10, height = 6,bg = "white")

print(img_path)
