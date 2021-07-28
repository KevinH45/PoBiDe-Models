import wikipedia, pickle, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from wikipedia.exceptions import DisambiguationError, PageError

class MisinformationDetector():
	def __init__(self):
		self.model = None
		self.tV = None

	def train(self):
		#This training method will not run. This method within the Github repository is merely to display the code
		
		raise NotImplementedError("This training method will not run. This method within the Github repository is merely to display the code")
		print ("Program Log: Training has begun.")
		
		self.data = pd.read_csv(r'train.csv')
		self.data = self.data.dropna(how='any',axis=0)

		self.tV = TfidfVectorizer(stop_words='english', max_df=0.7)

		
		lb = self.data.label
		x_train,x_test,y_train,y_test=train_test_split(self.data['text'], lb, test_size=0.2, shuffle=True, random_state=7)
		train=self.tV.fit_transform(x_train) 

		test=self.tV.transform(x_test)

		self.model=PassiveAggressiveClassifier(early_stopping=True, warm_start=True)
		self.model.fit(train,y_train)
		y_predict=self.model.predict(test)
		score=accuracy_score(y_test,y_predict)

		print ('Program Log: Model has been trained on the dataset.')
		print ("The accuracy is:",round(score*100,2))

		print (confusion_matrix(y_test,y_predict, labels=[1,0]))

	def save(self):
		#saving model
		if not self.model:
			raise TypeError('Model is of type None. Please define by training the model first.')
		filename = 'Pickled_Misinformation_PAC_Model'
		with open(filename, 'wb') as file:  
			pickle.dump(self.model, file)
		print ('Program Log: Model successfully saved.')
		
		#saving vectorizer
		if not self.tV:
			raise TypeError('Vectorizer is of type None. Please define by training the model first.')
		filename = 'Pickled_Misinformation_TFIDF_Vectorizer'
		with open(filename,'wb') as file:
			pickle.dump(self.tV,file)
		print ('Program Log: Vectorizer successfully saved.')

	def load(self):
		#Loading PAC model
		try:
			with open('Pickled_Misinformation_PAC_Model', 'rb') as file:  
				self.model = pickle.load(file)
			print ('Program Log: Model successfully loaded.')
		except FileNotFoundError:
			print ('Program Log: File not loaded because file was not found.')
		
		#Loading TFIDF Vectorizer
		try:
			with open('Pickled_Misinformation_TFIDF_Vectorizer', 'rb') as file:  
				self.tV = pickle.load(file)
			print ('Program Log: Vectorizer successfully loaded.')
		except FileNotFoundError:
			print ('Program Log: File not loaded because file was not found.')

	def think(self,string):
		if not self.model or not self.tV:
			raise TypeError('Model or Vectorizer is of type None. Please define by training or loading the model first.')
		print ('Program Log: Model has predicted',self.model.predict(self.tV.transform([string])))
		return(self.model.predict(self.tV.transform([string])))
	
	def search(self,term):
		try:
			return (wikipedia.summary(term))
		except PageError:
			return ("Page not found")
		except DisambiguationError:
			return ("Too many results for page or query too vague")
