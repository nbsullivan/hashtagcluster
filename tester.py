
import glob
import json
from pprint import pprint as pp
import tweetutils
import numpy as np
import scipy
from collections import Counter
import pandas as pd







if __name__ == '__main__':
	# testing play ground, is set up to work on the #howtoconfuseamillennial dataset.
	# make sure this reflects what is in etl.py's on_status method.

	# setting up list for tweets to go into this acts like tweetlistmaster in etl
	tweetlistmaster = []

	# prepare list for holding number of RTs removed and number of clusters per batch
	info_list = []

	# blank df for holding cluster size info
	sizes_df = pd.DataFrame()

	# base file path
	base = 'data/millennial/HowtoConfuseaMillennial_batch'


	# loop over batch numbers
	for k in range(0,24):

		# start a dictionary to store batch info, (total tweet numbers, RTs removed, number of clusters) 
		batch_info = {}

		fil = base + '{0}.txt'.format(k)

		print fil

		# open and load the file
		f = open(fil, 'r')
		tweet_batch = json.load(f)

		# remove retweets from batch
		tweet_batch_noRT = tweetutils.RT_removal(tweet_batch)

		# track number of RTs removed.
		RTremoved = len(tweet_batch) - len(tweet_batch_noRT)

		print RTremoved, ' Retweets removed'
		
		# append file data to full list

		tweetlistmaster = tweetlistmaster + tweet_batch_noRT

		# if you only want use the 1000 most recent tweets uncomment this, 
		# note this will make the Total_tweets number in the info_df incorrect
		if len(tweetlistmaster) > 1000:
			print "reducing master list size to 1000"
			reducingindex = len(tweetlistmaster) - 1000
			tweetlistmaster = tweetlistmaster[reducingindex:]


		print 'clustering on: ', len(tweetlistmaster), ' tweets'

		# vectorize tf-idf -> rp
		vect_tweets = tweetutils.tf_idf_rp_tweets(tweetlist = tweetlistmaster)

		# do not get to the point of exclusively individual tweet clusters
		max_n = int(len(tweetlistmaster) * .5)

		# do silhouette analysis on master list
		tweet_pred = tweetutils.silhouette_analysis(vect_tweets, max_n = max_n)

		# create cluter information on master list
		cluster_df = tweetutils.clusterinfo(tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)

		# get number of clusters
		n_clusters = max(cluster_df['cluster_pred'].unique()) + 1

		# total tweets
		total_tweets = (k+1)*100

		# build the dictionary of batch information
		batch_info['Total_tweets'] = total_tweets
		batch_info['RTs'] = RTremoved
		batch_info['Clusters'] = n_clusters

		# append batch_info to list
		info_list.append(batch_info)


		# write to file and grab cluster sizes df
		cluster_sizes_df = tweetutils.counts_to_file(cluster_df=cluster_df, base = base, batchnumber = k)

		# append sizes df
		sizes_df = sizes_df.append(cluster_sizes_df, ignore_index=True)

		# write cluster_df to file
		cluster_df.to_csv(base + 'pred'+ '{0}.csv'.format(k), encoding='utf-8')

	# create df of info_list
	info_df = pd.DataFrame(info_list)

	info_df['CumRT'] = info_df['RTs'].cumsum()

	info_df.to_csv(base + 'info.csv')

	sizes_df.to_csv(base + 'clustersize.csv')




