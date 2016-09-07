import glob
import json
from pprint import pprint as pp


def tweeet_clean(tweet = None):
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


if __name__ == '__main__':

	# this will grab the file name where data is stored
	for fil in glob.glob('data/*.txt'):
		# store scrubbed tweets into list
		tweetlist = []

		# somehow open up this and read the data?
		# this doesnt currently work
		f = open(fil, 'r')
		tweet_batch = json.load(f)

		# loop over tweets 
		for tweet in tweet_batch:
			# cleaning and appending tweets
			tweet_dict = tweeet_clean(tweet = tweet) 
			tweetlist.append(tweet_dict)

		f.close()

		# Create file buffer object f that points to the file clean
		print fil[5:]
		f = open('data/clean/' + fil[5:], 'w')
		json.dump(tweetlist, f)
		f.close()
	