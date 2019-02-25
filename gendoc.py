import os, sys
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter

def tokenize_file(filename):
    """Opens filename. Removes all punctuation and tokenizes by whitespace. Returns a list
       with the words in filename.
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
    """Tokenizes all files in foldername. Returns a dictionary with lists of tokenized
       files in the different subfolders.
    """
    
    subfolders = {} 
    files = []
    for _, dirs, _ in os.walk(foldername): 
        for directory in dirs:
            documents = tokenize_subfolder(directory, foldername)
            subfolders[directory] = documents
    
    return subfolders

def remove_duplicates(subfolders):
    duplicates = []
    subfolder1,subfolder2 = subfolders.keys()
    for i,(filename1,document1) in enumerate(subfolders[subfolder1]):
        for j,(filename2,document2) in enumerate(subfolders[subfolder2]):
            if document1 == document2:
                duplicates.append((filename1,filename2))
                del subfolders[subfolder1][i]
                del subfolders[subfolder2][j]             

    return subfolders,duplicates,subfolder1,subfolder2

def build_vocabulary(subfolders):
    """Puts all words of all documents in subfolders into one list. Counts the occurences
       of each word. Returns vocabulary, a list of two-tuples with words and their counts,
       in descending order.
    """
    all_words = []

    for subfolder in subfolders.values():
        for filename,document in subfolder:
            for word in document:
                all_words.append(word)
    
    vocabulary = Counter(all_words).most_common()
    
    return vocabulary

def make_vectors_dataframe(vocabulary,subfolders):
    """Counts occurences of words in each document. Makes a vector for each document and
       adds the wordcounts in the indices corresponding to the word's index in vocabulary.
       Makes a list of three-tuples, each containing the name of subfolder, the filename
       and the raw counts vector. Returns a dataframe vectors_df made from the list of 
       three-tuples.
    """
    vectors_raw_counts = []
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
            vectors_raw_counts.append((subfolder,filename,vector))
    vectors_df = pd.DataFrame(vectors_raw_counts)
    vectors_df.columns = ["subfolder","filename","vector"]
    
    return vectors_df

def tfidf(vectors_df):
    """Applies tfidf to the raw counts vectors. Returns vectors_df with tfidf vectors 
       instead of raw counts vectors.
    """
    transformer = TfidfTransformer()
    # Turns vectors into an np.array, and then splits them to lists again before reassigning the panda column.
    vector_block = np.stack(vectors_df['vector'])
    tfidf_vector = transformer.fit_transform(vector_block) # This returns a sparse compressed matrix
    tfidf_vector = tfidf_vector.toarray() 
    tfidf_vector = np.split(tfidf_vector,len(vectors_df['vector']))
    tfidf_vector = [np.squeeze(doc) for doc in tfidf_vector]
    vectors_df['vector'] = tfidf_vector
    
    return vectors_df

def svd(vectors_df,dimensionality):
    """Truncates matrix to dimensionality dimensions via singular value decomposition.
       Returns vectors_df with truncated vectors instead of raw counts/tf idf vectors.
    """
    truncator = TruncatedSVD(dimensionality)
    truncated_vector = truncator.fit_transform(np.stack(vectors_df['vector']))
    truncated_vector = np.split(truncated_vector,len(vectors_df['vector']))
    truncated_vector = [np.squeeze(doc) for doc in truncated_vector]
    vectors_df['vector'] = truncated_vector
    
    return vectors_df
      
def write_outputfile(vectors_df, outputfile):
    """Writes vectors_df to outputfile.
    """
    # Disables summary printing, to be able to do calculations on the whole matrix in 
    # outputfile.
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
    subfolders,duplicates,subfolder1,subfolder2 = remove_duplicates(subfolders)
    print("Following documents are duplicates and has been removed: ")
    for docs in duplicates:
        print(subfolder1,docs[0],subfolder2,docs[1])
    vocabulary = build_vocabulary(subfolders)
 
    if args.basedims:
        if args.basedims >= len(vocabulary):
            print("Error: The number of top dimensions must be smaller than the total raw counts dimensions ({}). {} >= {}.".format(len(vocabulary),args.basedims,len(vocabulary)))
            sys.exit()
    
    if args.basedims and args.svddims:
        if args.svddims >= args.basedims:
            print("Error: The number of dimensions to truncate to must be smaller than the raw counts dimensions ({}). {} >= {}.".format(args.basedims,args.svddims,args.basedims))
            sys.exit()
    
    if args.svddims:
        if args.svddims >= len(vocabulary):
            print("Error: The number of dimensions to truncate to must be smaller than the raw counts dimensions ({}). {} >= {}.".format(len(vocabulary),args.svddims,len(vocabulary)))
            sys.exit()
              
    if not args.basedims:
        print("Using full vocabulary.")
    else: 
        vocabulary = vocabulary[:args.basedims]
        vectors_df = make_vectors_dataframe(vocabulary,subfolders)
        print("Using only top {} terms by raw count.".format(args.basedims))
    
    print("Loading data from directory {}.".format(args.foldername))
    
    vectors_df = make_vectors_dataframe(vocabulary,subfolders)
 
    if args.tfidf:
        vectors_df = tfidf(vectors_df)
        print("Applying tf-idf to raw counts.")
        # If both tf-idf and TruncatedSVD is applied, the SVD is applied to the matrix 
        # containing tf-idf (which then gets overwritten) instead of the matrix containing
        # raw counts. 
        if args.svddims:                
            vectors_df = svd(vectors_df,args.svddims)
            print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    elif args.svddims:
        vectors_df = svd(vectors_df,args.svddims)
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    
    write_outputfile(vectors_df,args.outputfile)

    print("Writing matrix to {}.".format(args.outputfile))