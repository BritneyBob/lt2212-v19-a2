import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

def read_as_df(filename):
    vectorfile_df = pd.read_csv(filename)
    vectors = [doc[1:-1] for doc in vectorfile_df['vector']]
    vectors = [np.fromstring(doc,sep=' ') for doc in vectors]
    vectorfile_df['vector'] = vectors
    
    return vectorfile_df

def similarity_within_topic(vectorfile_df):
    grain = vectorfile_df[vectorfile_df["subfolder"]=="grain"]
    grain_vectors = [doc for doc in grain['vector']]
    cs_grain = cosine_similarity(grain_vectors)
    
    crude = vectorfile_df[vectorfile_df["subfolder"]=="crude"]
    crude_vectors = [doc for doc in crude['vector']]
    cs_crude = cosine_similarity(crude_vectors)
    
    return cs_grain,cs_crude   
    
def similarity_between_topics(vectorfile_df):
    grain = vectorfile_df[vectorfile_df["subfolder"]=="grain"]
    grain_vectors = [doc for doc in grain['vector']]
    
    crude = vectorfile_df[vectorfile_df["subfolder"]=="crude"]
    crude_vectors = [doc for doc in crude['vector']]
    
    cs_between = cosine_similarity(grain_vectors,crude_vectors)
    
    return cs_between
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
    parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

    args = parser.parse_args()
    
    df_vectorfile = read_as_df(args.vectorfile)
    cosine_similarity_1 = similarity_within_topic(vectorfile_df)

    print("Reading matrix from {}.".format(args.vectorfile))
