
import numpy as np
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

def vectorize_file(in_file, out_file):
    """Vectorize a JSON file."""
    fpi = open(in_file, 'r')
    tweets = json.load(fpi)
    fpi.close()

    vectorizer = CountVectorizer(min_df = 1)
    vect_tweets = vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweets))

    fpo = open(out_file, 'w')
    json.dump(vect_tweets.toarray().tolist(), fpo)
    fpo.close()

def main():
    inf = 'data/clean/HowtoConfuseaMillennial_batch0'
    outf = 'data/vectorized/HowtoConfuseaMillennial_batch0'
    vectorize_file(inf, outf)

if __name__ == '__main__':
    main()
