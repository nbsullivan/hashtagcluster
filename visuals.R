
# change working directory to your data file path.
setwd('/Users/nick/Documents/math503/project1/hashtagcluster/data/millennial')
library(ggplot2)
library(wordcloud2)

# get list of csvs.
files = list.files(pattern="*.csv")

for(fil in files){
  # load data
  df <- read.csv(fil)
  
  new_df = df[FALSE,]
  
  # lets only look at the top 50% words or users in a cluster
  for(k in c(min(df$Cluster):max(df$Cluster))){
    print(k)
    dfa <- subset(df, Cluster == k)
    print(quantile(dfa$Count, names = FALSE)[4])
    dfa <- subset(dfa, Count >= quantile(Count, names = FALSE, c(.75)))
    print(head(df))
    new_df <- rbind(new_df,dfa)
  }
  
  df <- new_df
  
  df$Cluster <- as.factor(df$Cluster)
  
  if(grepl('user',fil)){
    print('we have a user file')
    g <- ggplot(data = df, aes(User, color = Cluster, fill = Cluster))

    
  } else if(grepl('word', fil)){
    print('we have a word file')
    g <- ggplot(data = df, aes(Word, color = Cluster, fill = Cluster))
  }
  
  g + geom_bar(aes(weight = Count)) +
    ggtitle(fil) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    facet_wrap(~Cluster, scales="free_x")
  ggsave(paste(substr(fil, 1,nchar(fil)-4),'bar.pdf'))
  
}

