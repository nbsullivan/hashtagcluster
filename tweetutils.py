
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from pprint import pprint as pp
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import scipy
from scipy import sparse
from collections import Counter
import jsocket
import pandas as pd

def tweet_counter(tweet_text):
    # input tweet text and output dictionary with keys being tweet text and value being
    # a cout of occurances of tweet text
    tweet_freq = [tweet_text.count(i) for i in tweet_text]
    tweet_count = dict(zip(tweet_text, tweet_freq))

    return tweet_count


def clean_tweet(tweet):
    """input a tweet and get back a dictionary form of its relevant content.
    return tweet text, username, tweet_id, & geo-location (coordinates)"""
    tweet_dict = {}
    tweet_dict['text'] = tweet['text']
    tweet_dict['screen_name'] = tweet['user']['screen_name']
    tweet_dict['tweet_id'] = tweet['id']
    tweet_dict['timestamp'] = int(tweet['timestamp_ms'])


    # return tweets with geo-location
    if tweet['geo'] != None:
        tweet_dict['geo'] = tweet['geo']['coordinates']

    return tweet_dict

def vectorize_tweets(tweetlist, n_dimensions = None):
    """vectorized tweetlist inputs and outputs a sparse matrix representation"""
    vectorizer = CountVectorizer(min_df = 1, stop_words='english')
    vectorized_tweets = vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
    names = vectorizer.get_feature_names()
    return vectorized_tweets, names


def clusterinfo(tweetlistmaster = None, tweet_pred = None):
    """append tweet_pred to tweetmasterlist and return the dataframe"""
    tweet_df = pd.DataFrame(tweetlistmaster)

    tweet_df['cluster_pred'] = tweet_pred

    return tweet_df



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


    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    vect_tweets = vectorizer.fit_transform(tweets)

    fpo = open(out_file, 'w')
    json.dump(vect_tweets.toarray().tolist(), fpo)
    fpo.close()


def silhouette_analysis(vectorized_tweets, max_n = 20):
    # need code for silhouette analysis
    sil_scr_prev = 0
    brk = 0
    for n in range(2,max_n):
        print 'testing ', n, ' clusters'
        # cluster
        clf = MiniBatchKMeans(n_clusters=n)
        tweet_pred = clf.fit_predict(vectorized_tweets)
        # cluster silhouette scores
        silhouette_avg = silhouette_score(vectorized_tweets, tweet_pred)
        print 'Silhouette average ', silhouette_avg

        # determine number of centroids to use for batch
        if silhouette_avg <= sil_scr_prev:
            sil_n = n - 1
            sil_avg = sil_scr_prev
            brk = 1
        # break if previous silhoutte score is smaller
        if brk == 1:
            break
        sil_scr_prev = silhouette_avg
        sil_pred_prev = tweet_pred


    return sil_pred_prev

def counts_to_file(cluster_df, base, batchnumber):
    """for a cluster_json write it to a csv file, for analysis"""

    # initializing lists to store dictionary rows
    word_list = []
    user_list = []
    cluster_size_list = []

    n_clusters = max(cluster_df['cluster_pred'].unique()) + 1

    for k in range(0,n_clusters):

        cluster_info_dict = {}
        # subset dataframe to a single cluster.
        subset_df = cluster_df[cluster_df['cluster_pred'] == k]

        print 'size of cluster ', k, ':', len(subset_df.index)

        # store data on cluster
        cluster_info_dict['batch'] = batchnumber
        cluster_info_dict['cluster_number'] = k
        cluster_info_dict['size'] = len(subset_df.index)

        # append to cluster size list
        cluster_size_list.append(cluster_info_dict)

        # bag of words representation of tweets from cluster
        word_count_dict = dict(Counter(" ".join(subset_df["text"]).split()))

        for key in word_count_dict:
            row = {}
            row['Word'] = key
            row['Count'] = word_count_dict[key]
            row['Cluster'] = k
            word_list.append(row)

        # user count for cluster
        user_count_dict = dict(Counter(" ".join(subset_df["screen_name"]).split()))

        for key in user_count_dict:
            row = {}
            row['User'] = key
            row['Count'] = user_count_dict[key]
            row['Cluster'] = k
            user_list.append(row)

    # put into df for easy writing
    user_df = pd.DataFrame(user_list)
    words_df = pd.DataFrame(word_list)
    cluster_size_df = pd.DataFrame(cluster_size_list)

    # write to file
    user_df.to_csv(base + '{0}usercount.csv'.format(batchnumber), encoding='utf-8')
    words_df.to_csv(base + '{0}wordcount.csv'.format(batchnumber), encoding='utf-8')

    return cluster_size_df


def tf_idf_tweets(tweetlist):
    """use tf_idf tweetlist inputs and outputs a sparse matrix representation"""
    tfidfer = TfidfVectorizer(stop_words='english')
    tfidf_tweet = tfidfer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
    return tfidf_tweet


def hash_tweets(tweetlist):
    """hash tweetlist inputs and outputs a sparse matrix representation"""
    hasher = FeatureHasher(input_type = "string")
    hashed_tweets = hasher.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
    return hashed_tweets

def tf_idf_lsa_tweets(tweetlist):
    """Perform TF-IDF vectorization and then reduce the dimension using truncated
    SVD and normalize to the Euclidean unit ball.

    See http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py"""
    tfidfer = TfidfVectorizer(stop_words = 'english')
    tfidf_tweets = tfidfer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))

    # reduce dimenions to 1/3 of total features
    n_dim = tfidf_tweets.shape[1] / 3

    svd = TruncatedSVD(n_dim)
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)
    
    
    tfidf_tweets = lsa.fit_transform(tfidf_tweets)

    return tfidf_tweets

def tf_idf_pca_tweets(tweetlist, n_dim = 200):
    """Like tf_idf_lsa_tweets(), but uses PCA instead of TruncatedSVD."""
    mypca = PCA(n_components = n_dim)
    normalizer = Normalizer(copy = False)
    reducer = make_pipeline(mypca, normalizer)
    tfidfer = TfidfVectorizer(stop_words = 'english')
    tfidf_tweets = tfidfer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
    tfidf_tweets = reducer.fit_transform(tfidf_tweets.toarray())

    return sparse.csr_matrix(tfidf_tweets)

def RT_removal(tweetlistmaster):
    #
    RT_filter = []
    for n in tweetlistmaster:

        if n['text'][0:3] != 'RT ':
            RT_filter.append(n)

    return RT_filter


def RT_condensor(tweetlistmaster):
    #
    RT_filter = []
    for n in tweetlistmaster:

        if n['text'][0:3] == 'RT ':
            RT_filter.append(n)

        dd = dict((tweet["text"], tweet) for tweet in RT_filter).values()


    return dd



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
