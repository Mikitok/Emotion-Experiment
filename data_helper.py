#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import re
import jieba
import pickle
import codecs
import itertools
import pandas as pd
import numpy as np
import math
from gensim import corpora
from collections import Counter
from gensim.models import word2vec
from zhon.hanzi import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

def load_sina(textfilename,labelfilename):
    f = open(textfilename, 'rb')
    texts = pickle.load(f)
    f.close()
    f = open(labelfilename, 'rb')
    labels = pickle.load(f)
    f.close()
    return texts, labels

def clean_text(text):
    # 清除空格
    text=text.replace(' ','')
    #让文本只保留汉字
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"[%s]+" % punctuation, "", text)  # 去除标点
    return text

def segment_word(text):
    #使用jieba对文本进行分词（精准模式）
    text=jieba.cut(text)
    return text

def stopwordslist(filename):
    f=open(filename)
    stopwords = [line.strip() for line in f.readlines()]
    f.close()
    return stopwords

def remove_words(articles, stopwords):
    texts = []
    for text in articles:
        # 去掉数字和标点
        text = clean_text(text)
        # 分词
        text = segment_word(text)
        # 去掉停用词
        outstr = ''
        for word in text:
            if word not in stopwords:
                if word != '\t' and len(word) > 0:
                    outstr += word
                    outstr += " "
        text = outstr
        texts.append(text.split(' '))
    return texts

def clear_words(articles, wordvectordict):
    texts=[]
    for article in articles:
        for word in article:
            if word not in wordvectordict.keys():
                article.remove(word)
        texts.append(article)
    return texts

def tfidf_words(texts,stopwords,topN):
    articles = [' '.join(text) for text in texts]

    # 计算tfidf的值
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=stopwords)
    weights = tfidf.fit_transform(articles).toarray()   #weight[i][j]第i个文本中第j个词的权重
    word = tfidf.get_feature_names()

    textnum=weights.shape[0]
    # 选取每个文本中topN的词
    ind = np.argsort(-1*weights,axis=1,)  # 每一行为一个文本对应的tfidf最大的单词序号矩阵
    topwords=[]
    for i in range(textnum):
        topword=[word[ind[i][j]] for j in range(topN)]  #得到每个文本对应的topN个单词的词表
        topwords.append(topword)
    articles=[]
    for i in range(textnum):
        for word in texts[i]:
            if word not in topwords[i]:
                texts[i].remove(word)
        articles.append(texts[i])

    return tfidf, articles

def outside_embedding(embeddingpath):
    f=codecs.open(embeddingpath, "rb")
    wordvectordict = pickle.load(f)
    worddict = corpora.dictionary.Dictionary()
    worddict.doc2bow(wordvectordict.keys(), allow_update=True)
    wordindex = {v: k for k, v in worddict.items()}
    return wordindex, wordvectordict

def random_embedding(texts, wordsize):
    word_counts = Counter(itertools.chain(*texts))  # 统计单词和出现的频率，chain可以把一组迭代对象串联起来，形成一个更大的迭代器，Counter以字典的键值对形式存储，其中元素作为key，其计数作为value。
    vocabulary_inv = [word[0] for word in word_counts.most_common()]  # 按照出现次序由大到小取出单词。most_common返回一个TopN列表。如果n没有被指定，则返回所有元素。当多个元素计数值相同时，排列是无确定顺序的。
    wordindex = {word: index for index, word in enumerate(vocabulary_inv)}  # 给每个单词标上序号
    wordvectordict = {}
    for word in wordindex:
        wordvectordict[word] = np.random.uniform(-0.25, 0.25, wordsize)  # 将每个单词随机初始化为一个300维的词向量。numpy.random.uniform(low,high,size)，从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    return wordindex, wordvectordict

def word2vec_embedding(texts, modelpath, wordsize):
    if not os.path.exists(modelpath):
        word2vecmodel = word2vec.Word2Vec(texts, size=wordsize, min_count=2)
        word2vecmodel.save(modelpath)
    word2vecmodel = word2vec.Word2Vec.load(modelpath)

    worddict = corpora.dictionary.Dictionary()
    worddict.doc2bow(word2vecmodel.wv.vocab.keys(), allow_update=True)
    wordindex = {v: k for k, v in worddict.items()}  # 词语的索引，从0开始编号
    wordvectordict = {word: word2vecmodel[word] for word in wordindex.keys()}  # 词语的词向量
    if '' not in wordindex.keys():
        wordindex['']=len(wordindex.keys())
        wordvectordict['']=np.zeros(wordsize)
    return wordindex, wordvectordict


def padding_text(texts, mask, padding_word=0, maxlength=None):
    # 将所有句子变为统一长度maxlength，如果maxlength==None，则转为最长句子长度，padding_word：填充词
    if maxlength is None: # Train
        sequence_length = np.max(mask)
    else: # Prediction
        sequence_length = math.ceil(maxlength)

    padded_texts = []
    for i in range(len(texts)):
        text = texts[i]
        num_padding = sequence_length - len(text)

        if num_padding < 0: # Prediction: 太长删掉
            padded_text = text[0:sequence_length]
        else:
            for i in range(num_padding):
                text.append(padding_word)
            padded_text = text
        padded_texts.append(padded_text)
    return padded_texts



def doc2vec(texts, wordindex):
    textvec=[]
    for sentence in texts:
        s=[]
        for word in sentence:
            if word in wordindex.keys():
                s.append(wordindex[word])
        textvec.append(np.array(s))
    return textvec

def singlelabel(labels):
    emotion=[]
    for label in labels:
        emt = np.argmax([int(i) for i in label])
        emotion.append(emt)
    return emotion

def prepare_data(seqs, labels=None):
    lengths = [len(s) for s in seqs]
    if labels:
        labels = np.array(labels).astype('int32')
        return [seqs, labels, np.array(lengths).astype('int32')]
    # return [np.array(seqs), np.array(lengths).astype('int32')]
    return [seqs, np.array(lengths).astype('int32')]

def remove_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]

def load_data(path, n_words):
    with open(path, 'rb') as f:
        dataset_x, dataset_label= pickle.load(f)
        train_set_x, train_set_y = dataset_x[0], dataset_label[0]
        test_set_x, test_set_y = dataset_x[1], dataset_label[1]
    #remove unknown words
    train_set_x = [remove_unk(x, n_words) for x in train_set_x]
    test_set_x = [remove_unk(x, n_words) for x in test_set_x]

    return [train_set_x, train_set_y], [test_set_x, test_set_y]

if __name__ == "__main__":
    train_file = './data/sinanews/'
    stopwordsfile='./data/stopwords.txt'

    labels, headline, articles=load_sina(train_file)
    stopwords=stopwordslist(stopwordsfile)










