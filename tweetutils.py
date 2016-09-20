
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint as pp
from sklearn.cluster import KMeans
import numpy as np
import scipy
from collections import Counter
import jsocket


def clean_tweet(tweet):
    # input a tweet and get back a dictionary form of its relevant content.
    # return tweet text, username, tweet_id, & geo-location (coordinates)
    tweet_dict = {}
    tweet_dict['text'] = tweet['text']
    tweet_dict['screen_name'] = tweet['user']['screen_name']
    tweet_dict['tweet_id'] = tweet['id']
        
    # return tweets with geo-location
    if tweet['geo'] != None:
        tweet_dict['geo'] = tweet['geo']['coordinates']

    return tweet_dict

def vectorize_tweets(tweetlist):
    # vectorized tweetlist inputs and outputs a sparce matrix representation
    vectorizer = CountVectorizer(min_df = 1)
    vectorized_tweets = vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
    names = vectorizer.get_feature_names()
    return vectorized_tweets, names


def clusterinfo(n = 2, vectorized_tweets = None, names = None, tweetlistmaster = None, tweet_pred = None):
    # we want to subset the vectorized tweets based on tweet_pred, also create a list of dictionarys with word counts per cluster
    dict_list = []

    # put everything into the full_info dict
    full_info = {}

    # extract users and tweet_ids from tweetmasterlist
    userlist = [tweet["screen_name"] for tweet in tweetlistmaster]
    tweet_id = [tweet['tweet_id'] for tweet in tweetlistmaster]


    # loop over number of subsets
    for k in range(n):

        # prepare a dictionary for cluster information
        cluster_dict = {}

        # rows with label k.
        indexes = [i for i, x in enumerate(tweet_pred) if x == k]
        
        # subset vectorized tweets
        subset_words = vectorized_tweets[np.array(indexes)]

        # sum the subset into  a single vector
        total_vector = subset_words.sum(axis = 0)

        # note .tolist will return a nested list
        total_vector = total_vector.tolist()
        total_vector = total_vector[0];

        # convert the long to int for entries in total_vector
        total_vector = map(int, total_vector)

        # form dictionary of words.
        word_dict = dict(zip(names, total_vector))

        # reduce the size of the word dictionary
        reduc_word = {k:v for k,v in word_dict.items() if v != 0}



        # get unigue users and number of tweets from them.
        cluster_users = [userlist[i] for i in indexes]
        user_dict = dict(Counter(cluster_users))

        # get tweet ids users and number of tweets from them.
        tweet_id_list = [tweet_id[i] for i in indexes]

        # get together list of all tweets in the cluster including: well all of it.
        cluster_tweet_list = [tweetlistmaster[i] for i in indexes]

        for idx, item in enumerate(cluster_tweet_list):
            item['vector_representation'] = subset_words[idx]


        # store everything.
        cluster_dict["tweet_ids"] = tweet_id_list
        cluster_dict["tweetsize"] = len(tweet_id_list)
        cluster_dict["userscounts"] = user_dict
        cluster_dict["bagofwords"] = reduc_word
        cluster_dict['tweet_data'] = cluster_tweet_list
        # append to list
        dict_list.append(cluster_dict)

    full_info["Clusterlist"] = dict_list
    full_info["totaltweet"] = len(tweetlistmaster)


    return full_info


# def clean_tweet(tweet):
#     tdict = {}
#     tdict['text'] = tweet['text']
#     tdict['screen_name'] = tweet['user']['screen_name']
#     tdict['tweet_id'] = tweet['id']

#     if tweet['geo'] != None:
#         tdict['geo'] = tweet['geo']['coordinates']

#     return tdict

# def clean_tweets(in_file, out_file):
#     """Clean the tweets in in_file and output the cleaned tweets to out_file"""
#     fpi = open(in_file, 'r')
#     tweets = json.load(fpi)
#     fpi.close()

#     cleaned_tweets = map(clean_tweet, tweets)

#     fpo = open(out_file, 'w')
#     json.dump(cleaned_tweets, fpo)

# def vectorize_tweets(tweetlist):
#     twtext = tweet['text']
#     vectorizer = CountVectorizer(min_df = 1)
#     return vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))

def vectorize_file(in_files, out_file):
    """Vectorize a list of JSON files."""
    tweets = []

    for infile in in_files:
        fpi = open(infile, 'r')
        tweetf = json.load(fpi)
        fpi.close()
        tweets = tweets + map(lambda tweet: tweet['text'], tweetf)
    
    print tweets


    vectorizer = CountVectorizer(min_df = 1)
    vect_tweets = vectorizer.fit_transform(tweets)

    fpo = open(out_file, 'w')
    json.dump(vect_tweets.toarray().tolist(), fpo)
    fpo.close()

def silhouette_analysis(vectorized_tweets):
    # to be updated with matt and karls method for determining number of clusters.

    # for now use n = 3
    n = 3

    # send vectorized tweets to clustering algorithm
    clf = KMeans(n_clusters=n)
    tweet_pred = clf.fit_predict(vectorized_tweets)

    return n,tweet_pred

def cluster_to_port(clusterlist):
    # eventaully send cluster lists to a local port

    

    return None

# def main():
#     predir = 'data/clean/'
#     hashtag = 'HowtoConfuseaMillennial'
#     pre = predir + hashtag + '_batch'
#     post = '.txt'
#     infs = []
    
#     for k in xrange(0, 10 + 1):
#         filepath = pre + str(k) + post
#         infs = infs + [filepath]
    
#     outf = 'data/vectorized/HowtoConfuseaMillennial.json'
#     vectorize_file(infs, outf)


