library(ggplot2)
library(wordcloud2)
library(Hmisc)


# change working directory to your data file path.
setwd('/Users/nick/Documents/math503/project1/hashtagcluster/data/results')

# start of new visuals
clustersize <- read.csv("#Brangelina_batchclustersize.csv")

# min max cluster size this looks kinda stupid.
cluster_reduced <- subset(clustersize, batch %in% seq(from = 0, to = 5500, by = 10))
ggplot(data = cluster_reduced, aes(x = batch, y = size)) +
  geom_point(alpha = .5) +
  stat_summary(fun.y = 'mean', geom = 'line', color = 'blue') +
  ggtitle("Min and Max size of clusters on reduced dataset") +
  stat_summary(fun.y = "min", geom = "line", color = "green") +
  stat_summary(fun.y = "max", geom = "line", color = "purple")

# 

batch_info <- read.csv("#Brangelina_batchinfo.csv")

batch_names <- c("#Brangelina_batchinfo.csv", '#debates_batchinfo.csv', '#Emmys_batchinfo.csv')

new_df <- batch_info[FALSE,]

for(name in batch_names){
  batch <- read.csv(name)
  batch$Dataset <- sub("_batchinfo.csv","", name)
  new_df <- rbind(new_df,batch)
}

fullbatch <- new_df
  
ggplot(data = fullbatch, aes(x = X, y = RTs/100)) +
  geom_point(alpha = .15) +
  stat_smooth(level = 0) +
  ggtitle("Percentages of Retweets found in batches") +
  ylab("% of Retweets in batch") +
  xlab("Batch number") +
  facet_wrap(~Dataset, scales="free_x")

ggsave("RTlevelsplot.pdf")
    
  
fullbatch$Clusters <- as.factor(fullbatch$Clusters)
ggplot(data = fullbatch, aes(Clusters)) +
  geom_bar() +
  ggtitle('Number of Clusters') +
  facet_wrap(~Dataset, scales = "free_x")
  
ggsave("ClusterBarsplot.pdf")


