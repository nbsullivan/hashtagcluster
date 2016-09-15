
import json
from sklearn.feature_extraction.text import CountVectorizer

def clean_tweet(tweet):
    tdict = {}
    tdict['text'] = tweet['text']
    tdict['screen_name'] = tweet['user']['screen_name']
    tdict['tweet_id'] = tweet['id']

    if tweet['geo'] != None:
        tdict['geo'] = tweet['geo']['coordinates']

    return tdict

def clean_tweets(in_file, out_file):
    """Clean the tweets in in_file and output the cleaned tweets to out_file"""
    fpi = open(in_file, 'r')
    tweets = json.load(fpi)
    fpi.close()

    cleaned_tweets = map(clean_tweet, tweets)

    fpo = open(out_file, 'w')
    json.dump(cleaned_tweets, fpo)

def vectorize_tweets(tweetlist):
    twtext = tweet['text']
    vectorizer = CountVectorizer(min_df = 1)
    return vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))

def main():
    clean_tweets('data/sampletweets.txt', 'data/cltweets.json')

if __name__ == '__main__':
    main()
