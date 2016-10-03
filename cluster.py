import json
import glob
import tweetutils
import shutil
import time

if __name__ == '__main__':
	
	# ongoing tweet collection list
	tweetlistmaster = []
	roundnumber = 1

	while True:
		# check for new files
		if len(glob.glob('data/current_data/*.txt')) > 0:
			for fil in glob.glob('data/current_data/*.txt'):

				time.sleep(.5)
				print "loading ", fil
				f = open(fil, 'r')
				tweet_batch = json.load(f)

				# remove retweets
				tweet_batch_noRT = tweetutils.RT_removal(tweet_batch)

				print len(tweet_batch) - len(tweet_batch_noRT), ' Retweets removed'

				# append file data to full list
				tweetlistmaster = tweetlistmaster + tweet_batch_noRT

				print 'clustering on: ', len(tweetlistmaster), ' tweets'

				tweetlistmaster = tweetlistmaster + tweet_batch

				# movefiles once loaded
				newpath = "data/processed_data/" + fil[18:]
				shutil.move(fil, newpath)


			# vectorize tf-idf -> lsa
			vect_tweets = tweetutils.tf_idf_lsa_tweets(tweetlist = tweetlistmaster, n_dim = 87)

			# do not get to the point of exclusively individual tweet clusters
			max_n = int(len(tweetlistmaster) * .5)

			# do silhouette analysis on master list
			tweet_pred = tweetutils.silhouette_analysis(vect_tweets, max_n = max_n)

			# create cluter information on master list
			cluster_df = tweetutils.clusterinfo(tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)

			# write to file
			tweetutils.counts_to_file(cluster_df=cluster_df, base = newpath[:-4], batchnumber = roundnumber)

			# write cluster_df to file
			cluster_df.to_csv(newpath[:-4] + 'pred'+ '{0}.csv'.format(roundnumber), encoding='utf-8')


			print "finished batch ", roundnumber, " total tweets clustering on ", len(tweetlistmaster)
			roundnumber = roundnumber + 1
