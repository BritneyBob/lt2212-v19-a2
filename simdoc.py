import os, sys
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

def similarity_within_topic(vectorfile_df):
    """Calculates, for each subfolder, the cosine similarity of every vector of subfolder
       with every other vector of the same subfolder. Calculates the average of the cosine
       similarities. Returns the average cosine similarity for both topics.
    """
    grain = vectorfile_df[vectorfile_df["subfolder"]=="grain"]
    grain_vectors = [doc for doc in grain['vector']]
    cs_grain = cosine_similarity(grain_vectors)
    cs_sum = 0
    for similarity in np.nditer(cs_grain):
        cs_sum += similarity
    average_cs_grain = cs_sum / (cs_grain.shape[0] * cs_grain.shape[1])
    
    crude = vectorfile_df[vectorfile_df["subfolder"]=="crude"]
    crude_vectors = [doc for doc in crude['vector']]
    cs_crude = cosine_similarity(crude_vectors)
    cs_sum = 0
    for similarity in np.nditer(cs_crude):
        cs_sum += similarity
    average_cs_crude = cs_sum / (cs_crude.shape[0] * cs_crude.shape[1])
    
    return average_cs_grain,average_cs_crude   
    
def similarity_between_topics(vectorfile_df):
    """Calculates the cosine similarity of every vector of subfolder
       with every vector of the other subfolder and vice versa. Calculates the average of
       the cosine similarities. Returns the average cosine similarity for the two 
       comparisons (although the two values are exactly the same).
    """
    grain = vectorfile_df[vectorfile_df["subfolder"]=="grain"]
    grain_vectors = [doc for doc in grain['vector']]
    
    crude = vectorfile_df[vectorfile_df["subfolder"]=="crude"]
    crude_vectors = [doc for doc in crude['vector']]
    
    cs_between_gc = cosine_similarity(grain_vectors,crude_vectors)
    cs_between_cg = cosine_similarity(crude_vectors,grain_vectors)
    
    cs_sum = 0
    for similarity in np.nditer(cs_between_gc):
        cs_sum += similarity
    average_cs_gc = cs_sum / (cs_between_gc.shape[0] * cs_between_gc.shape[1])
    
    cs_sum = 0
    for similarity in np.nditer(cs_between_cg):
        cs_sum += similarity
    average_cs_cg = cs_sum / (cs_between_cg.shape[0] * cs_between_cg.shape[1])
    
    return average_cs_gc, average_cs_cg
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
    parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

    args = parser.parse_args()
    
    vectorfile_df = read_as_df(args.vectorfile)
    cosine_similarity_within = similarity_within_topic(vectorfile_df)
    cosine_similarity_between = similarity_between_topics(vectorfile_df)
    
    print("The average cosine similarity between the documents in 'grain': ", cosine_similarity_within[0])
    print("The average cosine similarity between the documents in 'crude': ", cosine_similarity_within[1])
    print("The average cosine similarity between the documents in 'grain' and the documents in 'crude': ", cosine_similarity_between[0])
    print("The average cosine similarity between the documents in 'crude' and the documents in 'grain': ", cosine_similarity_between[1])

    print("Reading matrix from {}.".format(args.vectorfile))
