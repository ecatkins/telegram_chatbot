import numpy as np
import pandas as pd
import gensim
from gensim import utils, matutils, models
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from secret import *

def train_main():
	'''	Trains an sklearn logistic regresion model on the given 
		example set, utilising a pre-trained Doc2Vec model to 
		generate feature vectors for text
	'''

	# Reads the training examples
	examples = pd.read_csv('hardcode_examples.csv')

	# Loads the Doc2Vdec model
	doc_model = models.Doc2Vec.load(doc2vec_location)

	# Initiates empty arrays for variables and labels
	doc_features = np.empty([0, doc_model.vector_size])
	labels = []

	# Initiaties a list to store the textual responses for later replaying
	unique_responses = []

	# For each example
	for index, row in examples.iterrows():
		
		# Extract the text and the ideal response
		text = row['Text']
		response = row['Response']

		# Create a feature vector for the text and add to feature array 
		vector = doc_model.infer_vector(utils.simple_preprocess(text), steps = 40)
		doc_features = np.concatenate((doc_features, [np.array(vector)]), axis = 0)

		# Add the label to that array and the text of the response to that array
		if response in unique_responses:
			label = unique_responses.index(response)
		else:
			unique_responses.append(response)
			label = len(unique_responses) - 1
		labels.append(label)

	labels = np.array(labels)	
	
	# Generate and fit logistic regression model
	model = LogisticRegression()
	model.fit(doc_features, labels)

	# Save to disk
	model_details = {'model':model, 'response_vector': unique_responses}
	joblib.dump(model_details, 'model.pkl')



if __name__ == '__main__':
	train_main()