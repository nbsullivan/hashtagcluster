import glob
import json


def tweeet_clean(tweet = None):
	# input a tweet and get back a dictionary form of its relevant concent.

	tweet_dict = tweet
	return tweet_dict



if __name__ == '__main__':

	print "hi matt"
	# this will grab the file name where data is stored
	for fil in glob.glob('data/*.txt'):
		# printing out the file name
		print fil

		# somehow open up this and read the data?
		# this doesnt currently work
		f = open(fil, 'r')
		tweet_batch = json.loads(f)
		print tweet_batch

	