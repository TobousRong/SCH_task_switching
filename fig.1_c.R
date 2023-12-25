library(ggplot2)
library(ggdendro)
library(ggridges)
library(reshape2)
library(sysfonts)
library(showtextdb)
library(showtext)
library(gridExtra)
library(cowplot)
library(ggpubr)
library(ggplot2)
library(ggradar)
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
install.packages("D:/devtools_2.4.5.zip", repos = NULL)
devtools::install_github("ricardo-bion/ggradar")
library(ggradar)
mydata <- read.csv("D:/taskswitch100/Hin.csv",1)
par(mar = c(0,0,0,0))
a <- ggradar(mydata,base.size = 3,axis.label.size = 8,grid.label.size =8,
             values.radar = c("0","-0.25","-0.4"),
             background.circle.colour = 'white',
             group.colours ="#999999",
             group.point.size = 3,
             grid.max = 0.4,grid.mid = 0.25,grid.min = 0,
             gridline.min.linetype = "solid",
             gridline.mid.colour = "#6B9BC3",
             grid.line.width = 1,
             legend.position = "bottom",
             plot.extent.x.sf = 1)
a
ggsave("Hin_Relative change.tiff",width=5,height = 4,path = "D:/taskswitch100")