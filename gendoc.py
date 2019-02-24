import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter

pd.set_option('display.max_colwidth', 20)

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

def tokenize_file(filename):
    """Opens filename. Removes all punctuation and tokenizes by whitespace.
       Returns a list with the words in filename.
    """
    with open(filename,'r') as f:
        text = f.read()

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokenized_document = re.split(r'\s+',text)
    tokenized_document = tokenized_document[:-1]
    
    return tokenized_document

def tokenize_subfolder(subfolder, foldername):
    """Tokenizes all files in subfolder. Returns a list with the tokenized files.
    """
    
    filepaths = []
    for root,_,filenames in os.walk(foldername + subfolder):
        for filename in filenames:
            filepaths.append((filename, os.path.join(root, filename)))
                
    tokenized_documents = []
    
    for filename,filepath in filepaths:
        tokenized_document = tokenize_file(filepath)
        tokenized_documents.append((filename, tokenized_document))
    
    return tokenized_documents
               
def tokenize_folder(foldername):
    """Tokenizes all files in foldername. Returns a dictionary with lists of 
       tokenized files in the different subfolders.
    """
    
    subfolders = {} 
    files = []
    for _, dirs, _ in os.walk(foldername): 
        for directory in dirs:
            documents = tokenize_subfolder(directory, foldername)
            subfolders[directory] = documents
    
    return subfolders

def build_vocabulary(subfolders):
    """Puts all words of all documents in subfolders into one list. Counts
       the occurences of each word. Returns vocabulary, a list of two-tuples
       with words and its counts, in descending order.
    """
    all_words = []

    for subfolder in subfolders.values():
        for filename,document in subfolder:
            for word in document:
                all_words.append(word)
    
    vocabulary = Counter(all_words).most_common()
    
    return vocabulary

def make_vectors_dataframe(vocabulary,subfolders):
    """Counts occurences of words in document. blablabla
    """
    vectors = []
    for subfolder,documents in subfolders.items():
        for filename,document in documents:
            counter_words = Counter(document).most_common()
            vector = np.zeros((len(vocabulary),), dtype=int)
            for word,count in counter_words:
                word_index = 0
                for wordcount in vocabulary:
                    if word in wordcount:
                        vector[word_index] = count 
                    word_index += 1
            vectors.append((subfolder,filename,vector))
    vectors_df = pd.DataFrame(vectors)
    vectors_df.columns = ["subfolder","filename","vector"]
    
    return vectors_df

def tfidf(vectors_df):
    transformer = TfidfTransformer()
    tfidf_vector = transformer.fit_transform(np.stack(vectors_df['vector']))
    tfidf_vector = tfidf_vector.toarray()
    tfidf_vector = np.split(tfidf_vector,len(vectors_df['vector']))
    tfidf_vector = [np.squeeze(doc) for doc in tfidf_vector]
    vectors_df['vector'] = tfidf_vector
    
    return vectors_df

def svd(vectors_df,dimensionality):
    truncator = TruncatedSVD(dimensionality)
    svd_vector = truncator.fit_transform(np.stack(vectors_df['vector']))
    svd_vector = np.split(svd_vector,len(vectors_df['vector']))
    svd_vector = [np.squeeze(doc) for doc in svd_vector]
    vectors_df['vector'] = svd_vector
    
    return vectors_df
      
def write_outputfile(vectors_df, outputfile):
    np.set_printoptions(threshold=np.nan)
    vectors_df.to_csv(outputfile, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate term-document matrix.")
    parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
    parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
    parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
    parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
    parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

    args = parser.parse_args()

    subfolders = tokenize_folder(args.foldername)
    vocabulary = build_vocabulary(subfolders)
    
    if not args.basedims:
        print("Using full vocabulary.")
    else:   
        vocabulary = vocabulary[:args.basedims]
        print("Using only top {} terms by raw count.".format(args.basedims))

    vectors_df = make_vectors_dataframe(vocabulary,subfolders)
    
    print("Loading data from directory {}.".format(args.foldername))
 
    if args.tfidf:
        vectors_df = tfidf(vectors_df)
        print("Applying tf-idf to raw counts.")

    if args.svddims:
        vectors_df = svd(vectors_df,args.svddims)
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    
    write_outputfile(vectors_df,args.outputfile)

    # THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
    # PARAMETERS.

    print("Writing matrix to {}.".format(args.outputfile))



