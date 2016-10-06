# change working directory to your data file path.
setwd('/Users/nick/Documents/math503/project1/hashtagcluster/data/millennial')
library(ggplot2)
library(wordcloud2)
library(Hmisc)

batch_info_df <- read.csv('#Emmys_batchinfo.csv')

ggplot(data = batch_info_df, aes(x = X, y = RTs/100)) +
  ggtitle('Retweet Percentage of Batches') +
  xlab('Batch Number') +
  ylab('Percentage of Retweets in Batch') +
  stat_smooth() +
  geom_point(alpha = .25)

ggplot(data = batch_info_df, aes(x = X, y = Clusters)) +
  ggtitle('Number of Clusters Per Batch') +
  xlab('Batch Number') +
  ylab('Number of Clusters') +
  ylim(0,10) +
  geom_point(alpha = .5)


cluster_info <- read.csv('#Emmys_batchclustersize.csv')

cluster_reduced <- subset(cluster_info, batch %in% seq(from = 0, to = 5500, by = 50))
ggplot(data = cluster_reduced, aes(x = batch, y = size)) +
  geom_point(alpha = .5) +
  stat_summary(fun.y = 'mean', geom = 'line', color = 'blue') +
  ggtitle("Min and Max size of clusters on reduced dataset") +
  stat_summary(fun.y = "min", geom = "line", color = "green") +
  stat_summary(fun.y = "max", geom = "line", color = "purple")
  
batch_words <- read.csv("HowtoConfuseaMillennial_batch18wordcount.csv")


batch0 <- subset(batch_words, Cluster == 0)
batch0$X <- NULL
batch0$Cluster <- NULL
batch0$word <- batch0$Word
batch0$freq <- batch0$Count
batch0$Word <- NULL
batch0$Count <- NULL

wordcloud2(batch0)

