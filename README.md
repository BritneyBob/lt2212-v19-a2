# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Britta Carlsson

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
      Average cosine similarity of documents
| File  |in "grain"|in "crude"|in "grain" compared to "crude" and vice versa|
|-------|:--------:|:--------:|:-------------------------------------------:|
| 1.txt | 0.3281   | 0.3720   | 0.3089                                      |
| 2.txt | 0.3387   | 0.3865   | 0.3201                                      |
| 3.txt | 0.0987   | 0.1068   | 0.0732                                      |
| 4.txt | 0.1153   | 0.1291   | 0.0882                                      |
| 5.txt | 0.4555   | 0.5021   | 0.4200                                      |
| 6.txt | 0.3300   | 0.3729   | 0.3097                                      |
| 7.txt | 0.4546   | 0.5023   | 0.4198                                      |
| 8.txt | 0.3396   | 0.3872   | 0.3207                                      |


### The hypothesis in your own words

I think the hypothesis was that the average cosine similarity will be higher within a topic than between topics. Maybe the hypothesis also was that the cosine similarity would be higher in "crude" than in "grain"? There shouldn't be a big difference in cosine similarities when only using the top 3000 terms, since words with very low counts doesn't say that much. The cosine similarity should be higher within topics when applying tf-idf, than when only using raw counts.

### Discussion of trends in results in light of the hypothesis

In all eight cases, the average cosine similarities are greater within the topics "grain" and "crude", than between the topics. The average cosine similarity is greater in "crude" than in "grain". Maybe "crude" mostly occurs together with "oil", and therefore the documents are more coherent than the documents containing "grain" (which could mean both "cereal" or for example to paint something), where the different interpretations maybe occurs with more variation. Since the cosine similarities within topics are higher than between topics, this measure could be a way to classify which of the topics an arbitrary document belongs to. There is no big difference between the cosine similarities when applying or not applying a vocabulary restriction. The cosine similarities were highest when applying TruncatedSVD to 100 dimensions, and lowest when applying tf-idf with the vocabulary restriction. Maybe this points to that the documents that are classsified by just one word that occurs in them actually are not that alike.