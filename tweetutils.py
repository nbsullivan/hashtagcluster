
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preporcessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from pprint import pprint as pp
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import scipy
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


def clusterinfo(n = 2, vectorized_tweets = None, names = None, tweetlistmaster = None, tweet_pred = None):
    """we want to subset the vectorized tweets based on tweet_pred
    also create a list of dictionaries with word counts per cluster"""
    dict_list = []

    # put everything into the full_info dict
    full_info = {}

    # extract users and tweet_ids from tweetmasterlist
    userlist = [tweet["screen_name"] for tweet in tweetlistmaster]
    tweet_id = [tweet['tweet_id'] for tweet in tweetlistmaster]
    tweet_text = [tweet['text'] for tweet in tweetlistmaster]

    # switch to np array for tweet texts
    nptweets = np.array(tweet_text)


    # loop over number of clusters
    for k in range(n):
        # prepare a dictionary for cluster information
        cluster_dict = {}

        # rows with label k.
        indexes = [i for i, x in enumerate(tweet_pred) if x == k]

        # extract tweet_texts for this cluster
        subset_tweets = nptweets[indexes]

        # join together tweets from list
        tweet_string = ' '.join(subset_tweets).split()

        # get counts of words
        word_count = [tweet_string.count(i) for i in tweet_string]

        # make into dict
        word_count_dict = dict(zip(tweet_string, word_count))


        # get unigue users and number of tweets from them.
        cluster_users = [userlist[i] for i in indexes]
        user_dict = dict(Counter(cluster_users))

        # get tweet ids users and number of tweets from them.
        tweet_id_list = [tweet_id[i] for i in indexes]

        # get together list of all tweets in the cluster including: well all of it.
        cluster_tweet_list = [tweetlistmaster[i] for i in indexes]

        # for idx, item in enumerate(cluster_tweet_list):
        #     item['vector_representation'] = subset_words[idx]


        # store everything.
        cluster_dict["tweet_ids"] = tweet_id_list
        cluster_dict["tweetsize"] = len(tweet_id_list)
        cluster_dict["userscounts"] = user_dict
        cluster_dict["bagofwords"] = word_count_dict
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


    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    vect_tweets = vectorizer.fit_transform(tweets)

    fpo = open(out_file, 'w')
    json.dump(vect_tweets.toarray().tolist(), fpo)
    fpo.close()


def silhouette_analysis(vectorized_tweets):
    # need code for silhouette analysis
    sil_scr_prev = 0
    brk = 0
    for n in range(2,20):
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

    if sil_n == None:
        sil_n = 10


    return sil_n, sil_pred_prev

def counts_to_file(cluster_json, base, batchnumber):
    """for a cluster_json write it to a csv file, for analysis"""

    # get lists of words and users
    word_list = []
    user_list = []

    # loop over the clusters
    for j in range(len(cluster_json['Clusterlist'])):

        # each instance of word or user is saved with a count and the cluster that it is part of
        word_row = {}
        user_row = {}

        # set up a word row
        for key in cluster_json['Clusterlist'][j]['bagofwords']:

            word_row["Cluster"] = j
            word_row["Word"] = key
            word_row["Count"] = cluster_json['Clusterlist'][j]['bagofwords'][key]

            word_list.append(word_row)

            word_row = {}

        for key in cluster_json['Clusterlist'][j]['userscounts']:

            user_row["Cluster"] = j
            user_row["User"] = key
            user_row["Count"] = cluster_json['Clusterlist'][j]['userscounts'][key]

            user_list.append(user_row)

            user_row = {}

    # put into  df for easy writing.
    user_df = pd.DataFrame(user_list)
    words_df = pd.DataFrame(word_list)

    # write to file
    user_df.to_csv(base + '{0}usercount.csv'.format(batchnumber), encoding='utf-8')
    words_df.to_csv(base + '{0}wordcount.csv'.format(batchnumber), encoding='utf-8')

    print user_df.head()
    print words_df.head()


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

def tf_idf_lsa_tweets(tweetlist, n_dim = 200):
    """Perform TF-IDF vectorization and then reduce the dimension using truncated
    SVD and normalize to the Euclidean unit ball.

    See http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py"""
    svd = TruncatedSVD(n_dim)
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)
    tfidfer = TfidfVectorizer(stop_words = 'english')
    tfidf_tweets = tfidfer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
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

    return tfidf_tweets



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
