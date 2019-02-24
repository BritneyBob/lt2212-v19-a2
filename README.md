# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Britta Carlsson

## Additional instructions

Document here additional command-line instructions or other details you
want us to know about running your code.

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
    # PARAMETERS.

## File naming convention

"1.txt": Contains a term-document matrix with no vocabulary restriction and no other transformations.
"2.txt": Contains a term-document matrix with a vocabulary restriction of 3000 top terms and no other transformations.
"3.txt": Contains a term-document matrix with no vocabulary restriction, but with tf-idf applied.
"4.txt": Contains a term-document matrix with a vocabulary restriction of 3000 top terms, and with tf-idf applied.
"5.txt": Contains a term-document matrix with no vocabulary restriction, but with truncated SVD applied to 100 dimensions.
"6.txt": Contains a term-document matrix with no vocabulary restriction, but with truncated SVD applied to 1000 dimensions.
"7.txt": Contains a term-document matrix with a vocabulary restriction of 3000 top terms, and with truncated SVD applied to 100 dimensions.
"8.txt": Contains a term-document matrix with a vocabulary restriction of 3000 top terms, and with truncated SVD applied to 1000 dimensions.

## Results and discussion

### Vocabulary restriction.

I chose to use the 3000 top terms by raw count. It had to be more than 1000 terms, to get the truncated SVD to work when applied to 1000 dimensions. I thought 3000 terms was significantly fewer than the 11 thousand something total terms in the documents, thereby hopefully avoiding words appearing only once (or a very few times).

### Result table
| File  | Grain within | Crude within | Between Grain and Crude | Between Crude and Grain |
|-------|:------------:|:------------:|:-----------------------:|:-----------------------:|
| 1.txt | 0.3317       | 0.3726       | 0.3110                  | 0.3110                  |
| 2.txt | 0.3425       | 0.3874       | 0.3224                  | 0.3224                  |
| 3.txt | 0.0997       | 0.1070       | 0.0738                  | 0.0738                  |
| 4.txt | 0.1168       | 0.1296       | 0.0890                  | 0.0890                  |
| 5.txt | 0.4600       | 0.5028       | 0.4225                  | 0.4225                  |
| 6.txt | 0.3328       | 0.3735       | 0.3117                  | 0.3117                  |
| 7.txt | 0.4594       | 0.5032       | 0.4226                  | 0.4226                  |
| 8.txt | 0.3232       | 0.3879       | 0.3228                  | 0.3228                  |


### The hypothesis in your own words
The average cosine similarity will be greater within a topic than between topics.

### Discussion of trends in results in light of the hypothesis

When eliminating duplicates, makes bias of smaller similarity than it really is (but actually the similarity gets higher???).

## Bonus answers

Would be better to not eliminate duplicates.
