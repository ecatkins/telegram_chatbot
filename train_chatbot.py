import numpy as np
import pandas as pd
import gensim

from gensim import utils, matutils, models

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from secret import *

def train_main():
	examples = pd.read_csv('hardcode_examples.csv')

	doc_model = models.Doc2Vec.load(doc2vec_location)

	doc_features = np.empty([0, doc_model.vector_size])
	labels = []

	unique_responses = []

	for index, row in examples.iterrows():
		text = row['Text']
		response = row['Response']
		vector = doc_model.infer_vector(utils.simple_preprocess(text), steps = 40)
		doc_features = np.concatenate((doc_features, [np.array(vector)]), axis = 0)

		if response in unique_responses:
			label = unique_responses.index(response)
		else:
			unique_responses.append(response)
			label = len(unique_responses) - 1
		labels.append(label)

	labels = np.array(labels)	
	model = LogisticRegression()

	model.fit(doc_features, labels)

	model_details = {'model':model, 'response_vector': unique_responses}

	joblib.dump(model_details, 'model.pkl')



if __name__ == '__main__':
	train_main()