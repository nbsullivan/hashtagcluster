import json
import glob
import tweetutils
import shutil
import time
import pandas as pd

if __name__ == '__main__':
	
	# ongoing tweet collection list
	tweetlistmaster = []
	roundnumber = 1

	# info list for RT and cluster number
	info_list = []
	k = 0
	sizes_df = pd.DataFrame()

	while True:

		# check for new files
		if len(glob.glob('data/current_data/*.txt')) > 0:

			# information dictionary
			batch_info = {}

			# hold number of RTs removed.
			RT_removed = 0

			for fil in glob.glob('data/current_data/*.txt'):

				# for each file increment k which holds the number of batch files opened.
				k = k + 1

				# pause so we do not accidently try and access a file that is being written to.
				time.sleep(.5)

				print "loading ", fil
				f = open(fil, 'r')
				tweet_batch = json.load(f)

				# remove retweets
				tweet_batch_noRT = tweetutils.RT_removal(tweet_batch)

				# increase number of retweets removed
				RT_removed = RT_removed + len(tweet_batch) - len(tweet_batch_noRT)

				# append file data to full list
				tweetlistmaster = tweetlistmaster + tweet_batch_noRT

				# movefiles once loaded
				newpath = "data/processed_data/" + fil[18:]
				shutil.move(fil, newpath)

			# new tweets have been loaded
			print 'clustering on: ', len(tweetlistmaster), ' tweets'

			# vectorize tf-idf -> lsa
			vect_tweets = tweetutils.tf_idf_lsa_tweets(tweetlist = tweetlistmaster)

			# do not get to the point of exclusively individual tweet clusters
			max_n = int(len(tweetlistmaster) * .5)

			# do silhouette analysis on master list
			tweet_pred = tweetutils.silhouette_analysis(vect_tweets, max_n = max_n)

			# create cluter information on master list
			cluster_df = tweetutils.clusterinfo(tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)

			# write to file and get sizes info
			cluster_sizes_df = tweetutils.counts_to_file(cluster_df=cluster_df, base = newpath[:-4], batchnumber = roundnumber)

			# append sizes df
			sizes_df = sizes_df.append(cluster_sizes_df, ignore_index=True)

			# write cluster_df to file
			cluster_df.to_csv(newpath[:-4] + 'pred'+ '{0}.csv'.format(roundnumber), encoding='utf-8')

			# get number of clusters
			n_clusters = max(cluster_df['cluster_pred'].unique()) + 1

			# build info dictionary
			batch_info['Total_tweets'] = k*100
			batch_info['RTs'] = RT_removed
			batch_info['Clusters'] = n_clusters

			info_list.append(batch_info)

			info_df = pd.DataFrame(info_list)

			info_df['CumRT'] = info_df['RTs'].cumsum()

			# write out data
			info_df.to_csv(newpath[:-4] + 'info.csv')
			sizes_df.to_csv(newpath[:-4] + 'clustersize.csv')


			print "finished batch ", roundnumber, " total tweets clustering on ", len(tweetlistmaster)
			roundnumber = roundnumber + 1


