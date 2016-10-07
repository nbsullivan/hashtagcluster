
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
from sklearn.cluster import SpectralClustering
from sklearn.random_projection import GaussianRandomProjection
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
    sil_scr_prev = -1
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

    # reduce dimenions to 1/4 of total features
    n_dim = tfidf_tweets.shape[1] / 4

    svd = TruncatedSVD(n_dim)
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)


    tfidf_tweets = lsa.fit_transform(tfidf_tweets)

    return tfidf_tweets


def tf_idf_rp_tweets(tweetlist):
    """Perform TF-IDF vectorization and use Gaussian random projection"""

    # tf-idf feature extraction
    tfidfer = TfidfVectorizer(stop_words = 'english')
    tfidf_tweets = tfidfer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))

    # random projection
    trans = GaussianRandomProjection()

    # try and use the default reduced dimension given by the Johnson-Lindenstrauss lemma
    try:
        trans = GaussianRandomProjection()
        RP_tweets = trans.fit_transform(tfidf_tweets)
    except:
        n_dim = tfidf_tweets.shape[1] / 4
        trans = GaussianRandomProjection(n_components = n_dim)
        RP_tweets = trans.fit_transform(tfidf_tweets)


    return RP_tweets

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

sid = SentimentIntensityAnalyzer()

def vader_sim(tweet1, tweet2, att_a, att_b):
    """Similarity measure based on the VADER sentiments of each tweet as
    specified in att_a and att_b."""
    s1 = sid.polarity_scores(tweet1)
    s2 = sid.polarity_scores(tweet2)

    m1 = np.exp(-np.abs(s1[att_a] - s2[att_b]))
    m2 = np.exp(-np.abs(s2[att_a] - s1[att_b]))

    return 0.5*(m1 + m2)

def vader_pos_sim(tweet1, tweet2):
    """Similarity measure based on the positive VADER sentiments of
    each tweet."""
    return vader_sim(tweet1, tweet2, 'pos', 'pos')


def vader_pos_neg_sim(tweet1, tweet2):
    """Similarity measure based on the positive and negative VADER sentiments of
    each tweet."""
    return vader_sim(tweet1, tweet2, 'pos', 'neg')

def vader_pos_neu_sim(tweet1, tweet2):
    """Similarity measure based on the positive and negative VADER sentiments of
    each tweet."""
    return vader_sim(tweet1, tweet2, 'pos', 'neu')

def spectral_vader(tweetlist, vectorized_tweets, sim_measure = vader_pos_sim, max_n = 20):
    """Perform spectral clustering with VADER and silhouette analysis."""
    affinity_matrix = vader_affinity_matrix(tweetlist, similarity = sim_measure)
    sil_scr_prev = -1
    brk = 0
    for n in range(2,max_n):
        print 'testing ', n, ' clusters'
        # cluster
        clf = SpectralClustering(n_clusters=n, affinity = 'precomputed')
        clf.fit(affinity_matrix)
        tweet_pred = clf.fit_predict(affinity_matrix)
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


def vader_affinity_matrix(tweetlist, similarity = vader_pos_neg_sim):
    """"Compute a real symmetric affinity matrix using a VADER similarity."""
    n_tweets = len(tweetlist)
    affinity_matrix = np.zeros([n_tweets, n_tweets])
    tweets = map(lambda tweet: tweet['text'], tweetlist)

    # compute the values below the diagonal
    for i in xrange(n_tweets):
        tweet1 = tweets[i]

        for j in xrange(i):
            tweet2 = tweets[j]
            affinity_matrix[i, j] = similarity(tweet1, tweet2)

    # the values above the diagonal are gven by
    # affinity_matrix[i, j]  == affinity_matrix[j, i]
    affinity_matrix = affinity_matrix + affinity_matrix.T

    ## fill in diagonal
    for i in xrange(n_tweets):
        tweet1 = tweets[i]
        affinity_matrix[i, i] = similarity(tweet1, tweet1)

    return affinity_matrix

def vader_cluster_sentiment(file_in, file_out):
    """Perform VADER sentiment analysis on a set of clusters in batch csv
    format."""
    batch_df = pd.read_csv(file_in)
    tweets = batch_df['text']
    compound = []
    pos = []
    neg = []
    neu = []

    for tweet in tweets:
        s = sid.polarity_scores(tweet)
        compound = compound + [s['compound']]
        pos = pos + [s['pos']]
        neg = neg + [s['neg']]
        neu = neu + [s['neu']]

    batch_df['VADERcompound'] = pd.Series(compound)
    batch_df['VADERpos'] = pd.Series(pos)
    batch_df['VADERneg'] = pd.Series(neg)
    batch_df['VADERneu'] = pd.Series(neu)

    batch_df.to_csv(file_out)

# vader_cluster_sentiment('data/millennial/HowtoConfuseaMillennial_batchpred7.csv', 'mill.csv')


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
