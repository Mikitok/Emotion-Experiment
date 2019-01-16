import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
import random
import numpy as np
import sys
import os 

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('made directory %s' % path)
make_dir("parsed_data")

def text_preprocessing(datasets):
	dataset_text=[]
	dataset_label=[]
	for file in datasets:
		lines=[]
		labels=[]
		with open(file) as f:
			for l in f:
				try:
					words=re.split('\s|-',l.lower().split("|||")[0].strip()) #得到单词
					words=words
					label=int(l.lower().split("|||")[1].strip())	#得到标签
					lines+=[words]
					labels+=[label]
				except:
					continue
		dataset_text+=[lines]
		dataset_label+=[labels]
	return dataset_text, dataset_label

#insert words of a file
def insert_word(dataset, all_words):
	for lines in dataset:
		for l in lines:
			all_words+=l

#convert words to numbers
def convert_words_to_number(dataset_text, dataset_label, common_word):
	transformed_text=[]
	transformed_label=[]
	for lines, labels in zip(dataset_text, dataset_label):
		new_x=[]
		new_label=[]
		for l, label in zip(lines,labels):
			words=[common_word[w] if w in common_word else 1 for w in l]	
			
			new_x+=[words]
			new_label+=[label]

		transformed_text+=[new_x]
		transformed_label+=[new_label]
	return transformed_text, transformed_label

if __name__ == "__main__":
	#take out frequent words. 0 not a word, 1 unknown word, 2-... words

	#dataset file
	#datasets=['sst_data/sent+phrase.binary.clean.train', 'sst_data/raw.clean.dev', 'sst_data/raw.clean.test']
	filename='apparel'
	datasets=['sst_data/'+filename+'_trn', 'sst_data/'+filename+'_dev', 'sst_data/'+filename+'_tst']
	#preprocess the texts
	dataset_text, dataset_label=text_preprocessing(datasets)
	#insert all words
	all_words=[]
	insert_word(dataset_text, all_words)

	#obtain frequent words
	counter=collections.Counter(all_words)
	vocab=len(counter)
	vocab_size=vocab-2	
	common_word=dict(counter.most_common(vocab_size))
	print(len(common_word))

	#number them
	c=2	#从2开始是因为开头和结尾都添加了符号
	for key in common_word:
		common_word[key]=c
		c+=1
	#write out filtering training test data 将文中的单词转为序号
	transformed_text, transformed_label= convert_words_to_number(dataset_text, dataset_label, common_word)

	pickle.dump(((transformed_text, transformed_label)), open('parsed_data/'+filename+'_dataset', 'wb'))

	glove_filename="./glove/glove.6B.300d.txt"
	glove_dict={}
	i=0
	with open(glove_filename, 'r', encoding='UTF-8') as f:
		for line in f:

			i=i+1
			line = line.strip().split(' ')
			word = line[0]
			embedding = [float(x) for x in line[1:]]
			glove_dict[word]=embedding
	print(i)
	word2vec=[np.random.normal(0, 0.1, 300).tolist(), np.random.normal(0, 0.1, 300).tolist()]
	missing=0
	for number, word in sorted(zip(common_word.values(), common_word.keys())):
		try:
			word2vec.append(glove_dict[word])
		except KeyError: 	
			word2vec.append(np.random.normal(0, 0.1, 300).tolist())
			missing+=1
		#print(number)
		print(len(word2vec))
	pickle.dump(word2vec, open(filename+'_vectors', 'wb'))
	print(missing)
	print(np.array(word2vec).shape)
	#create embedding vector matrix
"""	word_vectors = KeyedVectors.load_word2vec_format('../vectors.gz', binary=True)
	word2vec=[np.random.normal(0, 0.1, 300).tolist(), np.random.normal(0, 0.1, 300).tolist()]
	for number, word in sorted(zip(common_word.values(), common_word.keys())):
		try:
			print(type(word_vectors.word_vec(word)))
			word2vec.append(word_vectors.word_vec(word).tolist())
		except KeyError: 
			print(word+ " not found")
			word2vec.append(np.random.normal(0, 0.1, 300).tolist())
	pickle.dump(word2vec, open(sys.argv[1]+'_vectors', 'wb'))
	print(len(word2vec))"""
