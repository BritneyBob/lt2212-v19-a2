import os, sys # Behövs de här?
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

def read_as_df(filename):
    """Reads filename as a dataframe. Removes brackets from all vectors. Converts vectors 
       from strings to arrays. Returns the dataframe vectorfile_df. Returns a list with the
       words in filename.
    """
    vectorfile_df = pd.read_csv(filename)
    vectors = [doc[1:-1] for doc in vectorfile_df['vector']]
    vectors = [np.fromstring(doc,sep=' ') for doc in vectors]
    vectorfile_df['vector'] = vectors
    
    return vectorfile_df

def similarity_within_topic(vectorfile_df,topic_name):
    """Calculates, for each subfolder, the cosine similarity of every vector of topic
       with every other vector of the same topic. Calculates the average of the cosine
       similarities. Returns the average cosine similarity for both topics.
    """
    topic = vectorfile_df[vectorfile_df["subfolder"]==topic_name]
    topic_vectors = [doc for doc in topic['vector']]
    cs_topic = cosine_similarity(topic_vectors)
    cs_sum = 0
    for similarity in np.nditer(cs_topic):
        cs_sum += similarity
    average_cs_topic = cs_sum / (cs_topic.shape[0] * cs_topic.shape[1])
    
    return average_cs_topic   
    
def similarity_between_topics(vectorfile_df,topic1_name,topic2_name):
    """Calculates the cosine similarity of every vector of subfolder
       with every vector of the other subfolder and vice versa. Calculates the average of
       the cosine similarities. Returns the average cosine similarity for the two 
       comparisons (although the two values are exactly the same).
    """
    topic1 = vectorfile_df[vectorfile_df["subfolder"]==topic1_name]
    topic1_vectors = [doc for doc in topic1['vector']]
    
    topic2 = vectorfile_df[vectorfile_df["subfolder"]==topic2_name]
    topic2_vectors = [doc for doc in topic2['vector']]
    
    cs_between_1_2 = cosine_similarity(topic1_vectors,topic2_vectors)
    cs_between_2_1 = cosine_similarity(topic2_vectors,topic1_vectors)
    
    cs_sum = 0
    for similarity in np.nditer(cs_between_1_2):
        cs_sum += similarity
    average_cs_topic1_topic2 = cs_sum / (cs_between_1_2.shape[0] * cs_between_1_2.shape[1])
    
    cs_sum = 0
    for similarity in np.nditer(cs_between_2_1):
        cs_sum += similarity
    average_cs_topic2_topic1 = cs_sum / (cs_between_2_1.shape[0] * cs_between_2_1.shape[1])
    
    return average_cs_topic1_topic2, average_cs_topic2_topic1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
    parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

    args = parser.parse_args()
    
    vectorfile_df = read_as_df(args.vectorfile)
    cosine_similarity_within_grain = similarity_within_topic(vectorfile_df,'grain')
    cosine_similarity_within_crude = similarity_within_topic(vectorfile_df,'crude')
    cosine_similarity_between = similarity_between_topics(vectorfile_df,'grain','crude')
    
    print("The average cosine similarity between the documents in 'grain': ", cosine_similarity_within_grain)
    print("The average cosine similarity between the documents in 'crude': ", cosine_similarity_within_crude)
    print("The average cosine similarity between the documents in 'grain' and the documents in 'crude': ", cosine_similarity_between[0])
    print("The average cosine similarity between the documents in 'crude' and the documents in 'grain': ", cosine_similarity_between[1])

    print("Reading matrix from {}.".format(args.vectorfile))
