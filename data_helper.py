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
    return labels, texts

def load_semeval(filename):
    labels = []
    texts = []
    with open(filename, 'r') as f:
        sentences = f.readlines()
        for line in sentences:
            line = line.split('\t')
            emt = np.array(re.split('[\s:]', line[1]))
            emt = emt[[3, 5, 7, 9, 11, 13]]  # 取出每行的emotion对应的label，分别是anger disgust fear joy sad surprise，需要rank的时候加上all 1，再归一化
            labels.append(emt)
            text = re.sub(r"\n", "", line[2])
            texts.append(text.split())
    return labels, texts

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

def out_embedding(embeddingpath):
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
        word2vecmodel = word2vec.Word2Vec(texts, size=wordsize, min_count=1)
        word2vecmodel.save(modelpath)
    # word2vecmodel = word2vec.Word2Vec(texts, size=wordsize, min_count=1)
    word2vecmodel = word2vec.Word2Vec.load(modelpath)

    worddict = corpora.dictionary.Dictionary()
    worddict.doc2bow(word2vecmodel.wv.vocab.keys(), allow_update=True)
    wordindex = {v: k for k, v in worddict.items()}  # 词语的索引，从0开始编号
    wordvectordict = {word: word2vecmodel[word] for word in wordindex.keys()}  # 词语的词向量
    if '' not in wordindex.keys():
        wordindex['']=len(wordindex.keys())
        wordvectordict['']=np.zeros(wordsize)
    return wordindex, wordvectordict

def pad_sentences(sentences, padding_word="", maxlength=None):
	# 将所有句子变为统一长度forced_sequence_length，如果forced_sequence_length==None，则转为最长句子长度，padding_word：填充词
	if maxlength is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		#logging.critical('This is prediction, reading the trained sequence length')
		sequence_length = maxlength
	#logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: 太长删掉
			padded_sentence = sentence[0:sequence_length]
		else:	# Train：不够用padding_word补足
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def doc2vec(texts, wordindex):
    textvec=[]
    for sentence in texts:
        s=[]
        for word in sentence:
            if word in wordindex.keys():
                s.append(wordindex[word])
            # else:
            #     s.append(wordindex[''])
        textvec.append(np.array(s))
    # return np.array(textvec)
    return textvec

def onehot(labels):
    emotion=[]
    for label in labels:
        emt = np.argmax([int(i) for i in label])
        emotion.append(emt)
    # num_labels = len(label)
    # labels = [i for i in range(num_labels)]
    # one_hot = np.zeros((num_labels, num_labels), int)
    # np.fill_diagonal(one_hot, 1)
    # label_dict = dict(zip(labels, one_hot))
    # y_raw = [label_dict[y] for y in emotion]
    # y = np.array(y_raw)

    return emotion

def singlelabel(labels):
    emotion=[]
    for label in labels:
        emt = np.argmax([int(i) for i in label])
        emotion.append(emt)
    return emotion

def batch_iter(data, batch_size, num_epochs, shuffle=True): #shuffle：打乱
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))# 以data_size为终点，起点取默认值0，步长取默认值1生成随机数并打乱
			shuffled_data = data[shuffle_indices]	#将所有数据打乱
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]	#截出一个epoch下的一个batch

if __name__ == "__main__":
    train_file = './data/sinanews/'
    stopwordsfile='./data/stopwords.txt'

    labels, headline, articles=load_sina(train_file)
    stopwords=stopwordslist(stopwordsfile)










