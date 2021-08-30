import csv
from gensim import corpora
from gensim import models
from collections import defaultdict
from pprint import pprint
import re
import numpy as np

statement_data = []

with open(r'/Users/mertbayturk/Desktop/TexasLastWords/ls.csv',encoding="utf8", errors='ignore') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    for row in readCSV:
        isMale = 0
        try:
            isMale = int(float(row[18]))
        except ValueError:
            isMale = 0

        if isMale > 0:
            statement_data.append([row[19], "Male"])
        else:
            statement_data.append([row[19], "Female"])

stoplist = set('for a of the and to in on at'.split())

texts = [[word for word in re.sub('\W+', ' ', (document[0].lower())).split() if word not in stoplist]
         for document in statement_data]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

index = int(input("Which Index of the Data do you Want to Analyze? "))

new_doc1 = re.sub('\W+', ' ', (statement_data[index][0].lower()))
new_vec1 = dictionary.doc2bow(new_doc1.lower().split())

print("Gender: " + statement_data[index][1] + "\n")
print("Statement: " + statement_data[index][0] + "\n")
print("Dictionary Matched Vectors:")
print(new_vec1)


new_doc = input("What Statement do you Want to Analyze? ")
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec_x = dictionary.doc2idx(new_doc.lower().split(), unknown_word_index=-1)

print("Dictionary Matched Vectors: ")
print(new_vec)
print("Single Word Vector Analysis: ")
print(new_vec_x)

from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count

model = Word2Vec(texts, min_count=0, workers=cpu_count())
model.train(texts, total_examples=model.corpus_count, epochs=1)
print("Most Similar Words in Dictionary to Your Inputed Statement: ")
print(model.wv.most_similar(new_doc.lower().split()),"\n")

import gensim

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary)

print("Latent Dirichlet Allocation (LDA) Topic Analysis: ")
for i,topic in enumerate(lda.print_topics(num_topics=100, num_words=15)):
    words = topic[1].split("+")
    print("Statement ", i , words,"\n")

# Visualize LDA in iPython (CoLab, Jupyter, etc.)
# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# pyLDAvis.gensim.prepare(lda, corpus, dictionary)


# Create the TF-IDF model
# tfidf = models.TfidfModel(corpus, smartirs='ntc')
#
# for doc in tfidf[corpus]:
#     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
#
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#
# model = Doc2Vec(dictionary, vector_size=5, window=2, min_count=1, workers=4)
#
# sim = model.n_similarity(new_doc.lower().split())
#
# print("{:.4f}".format(sim))
