# Top2Vec: BERT + HDBSCAN (https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)
# use SPECTER model https://arxiv.org/abs/2004.07180 (this model was trained on scientific papers)

from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import umap
import hdbscan
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def make_clusters(data, neighbors=15, components=5, cluster_size=5, save_embeddings=False):
    """
    input: data should be an array of strings or matrixes of floats
           ex: data[0]==str, or data[0][0]==np.float
    returns: HDBSCAN object
    """
    try:
        if type(data[0])==str:
            model = SentenceTransformer('allenai-specter')
            embeddings = model.encode(data, show_progress_bar=True)
            if save_embeddings:
                print('saving embeddings')
                ts=str(time.time())
                np.savetxt('embeddings'+ts+'.txt', embeddings, delimiter=',')
        elif type(data[0][0])==np.float64:
            embeddings=data
    except Exception as e: 
        print(e)
        return
    
    umap_embeddings = umap.UMAP(n_neighbors=neighbors, 
                                n_components=components, 
                                metric='cosine').fit_transform(embeddings)

    cluster = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                              metric='euclidean',                      
                              cluster_selection_method='eom').fit(umap_embeddings)
    return cluster

def get_topics(cluster, abstracts, n_words=20):
    """
    input: 
        cluster- HDBSCAN object created by make_clusters(), 
        abstracts - array of str
        n_words - number of top words to return for each topic
    returns:
        top_n_words - dictionary containing top words for each topic as key. topic -1 are outliers (do not belong to any topic)
        topic_sizes - pandas dataframe containing the number of aricles per topic
    """
    
    docs_df = pd.DataFrame(abstracts, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})


    def c_tf_idf(documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(abstracts))

    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                         .Doc
                         .count()
                         .reset_index()
                         .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                         .sort_values("Size", ascending=False))
        return topic_sizes
    
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=n_words)
    topic_sizes = extract_topic_sizes(docs_df);
    
    return top_n_words, topic_sizes

def drawCloud(text,max_words=350,width=1000,height=600,figsize=(15,15),raw=True,title=""):
    wc = WordCloud(background_color="white", 
                   max_words=max_words, 
                   width=width, 
                   height=height, 
                   random_state=1)
    if raw:
        wc.generate(text)
    else:    
        wc.generate_from_frequencies(text)
    
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.axis("off")
    cloud=plt.imshow(wc)
    return cloud


def getTFIDF(text,top_n=15,maxdf=0.9):
    vectorizer = TfidfVectorizer(stop_words='english',min_df=0.2,max_df=maxdf,ngram_range=(1,3),strip_accents='unicode')
    X = vectorizer.fit_transform(text)
#     print(X.shape)
    
    feature_names=vectorizer.get_feature_names()
    words_scores=[]
    for col in X.nonzero()[1]:
        words_scores.append((feature_names[col],X[0, col]))
    #     print(feature_names[col], ' - ', X[0, col])

    words_scores=list(set(words_scores))
    word_scores_sorted=defaultdict(str)
    for item in sorted(words_scores, key = lambda x: -x[1])[:top_n]:
        word_scores_sorted[item[0]]=item[1]
#         print(item)
    return word_scores_sorted