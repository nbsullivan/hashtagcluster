library(ggplot2)
library(wordcloud2)
library(Hmisc)
library(plyr)


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


# word cloud demostration on #emmy batch 150

emmy_wc <- read.csv("emmybatchpreds/#Emmys_batch150wordcount.csv")

new_df = emmy_wc[FALSE,]

# lets only look at the top 75% words or users in a cluster
for(k in c(min(emmy_wc$Cluster):max(emmy_wc$Cluster))){
  print(k)
  print(c(min(emmy_wc$Cluster):max(emmy_wc$Cluster)))
  
  # subset the dataframe to a single cluster group
  dfa <- subset(emmy_wc, Cluster == k)

  # for plotting purposes only take the top 75% used words or users
  dfa <- subset(dfa, Count >= quantile(Count, names = FALSE, c(.75)))

  # rename headers for wordcloud2
  dfa <- rename(dfa, c('Count'='freq', 'Word'='word'))
  
  # remove extras
  dfa$X <-  NULL
  
  # wordcloud2 is a bit of a punk and needs proper order of column.
  dfa <- dfa[c(3,2,1)]
  
  print(head(dfa))
  
  # combind dfs back together
  new_df <- rbind(new_df,dfa)
}

emmy_wc <- new_df

# wordcloud2 cannot facet.
emmy_0 <- subset(emmy_wc, Cluster == 0)
emmy_1 <- subset(emmy_wc, Cluster == 1)
emmy_2 <- subset(emmy_wc, Cluster == 2)
emmy_3 <- subset(emmy_wc, Cluster == 3)

wordcloud2(emmy_0)
wordcloud2(emmy_1)
wordcloud2(emmy_2)
wordcloud2(emmy_3)
