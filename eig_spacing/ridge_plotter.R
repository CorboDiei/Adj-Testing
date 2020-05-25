library(ggplot2)
library(ggridges)
library(tidyverse)
library(bigmemory)
setwd('C:/Users/David Corbo/Desktop/Research/Atom_Adj/eig_spacing')

x <- read.csv(file='alpha_df.csv', header=TRUE)


ggplot(x, aes(x=d, y=as.factor(l), length=length(x))) + 
  geom_density_ridges(scale=80, size=.1, rel_min_height=.1) +
  theme_ridges(font_size=70) +
  scale_x_continuous(limits = c(0, .01), expand = c(0,0)) +
  scale_y_discrete(
    breaks = c(0, 100, 200, 300, 400, 500, 600)
  ) +
  labs(
    x = "Eigenvalue Spacing",
    y = "Eigenvalue Pair #",
    title = "Alpha Helix Eigenvalue Spacings"
  ) +
  coord_cartesian(clip="off")
