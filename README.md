# hashtagcluster

To Run:

1. Make sure that you have the required packages:
-Anaconda 2.7
-tweepy

2. Save the JSON auth file in the directory below this one.


Live Clustering:
3. find something that is trending on twitter

4. change the variable hashtag on line 83 of etl.py

5. start etl.py

6. start cluster.py

7. result files will show up in data/processed_data

Non-Live Clustering (tester.py):
3. Change the base file directory to location of data set (line 32)

4. Change the range of the for loop to the number of batch files (line 34)

5. result files will show up in same folder as batch files