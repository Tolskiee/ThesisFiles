#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import neccessary dependencies
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import joblib


# # Import the data

# In[2]:


# training data
train = pd.read_csv(r".\hatespeech\train.csv", nrows=10000)

# testing data
test = pd.read_csv(r".\hatespeech\test.csv", nrows=10000)




# # Clean the data of null values

# In[3]:


# drop rows with null values
train.dropna()


# In[4]:


#check for null values in train
train.isnull().sum()


# In[5]:


# check for 0 values in train
sum(train["label"] == 0)


# In[6]:


# check for 1 values in train
sum(train["label"] == 1)


# # Clean the data of unwated Text and Characters

# In[7]:


# remove special characters using the regular expression library

import re

#set up punctuations we want to be replaced

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")


# In[8]:


import preprocessor as p

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_tweets(df):
    tempArr = []
    for line in df:
        # send to tweet_processor
        tmpL = p.clean(line)
        # remove everything except letters and digits
        tmpL = re.sub(r'[^a-zA-Z0-9\s]', ' ', tmpL)
        # convert to lowercase
        tmpL = tmpL.lower()
        tempArr.append(tmpL)
    return tempArr


# In[9]:


# clean training data
train_tweet = clean_tweets(train["text"])
train_tweet = pd.DataFrame(train_tweet)


# In[10]:


# append cleaned tweets to the training data
train["clean_tweet"] = train_tweet


# In[11]:


sum(train['clean_tweet'] == '')


# In[12]:


#df['Another'] = df['Another'].replace('', np.nan)
#replace all empty spaces with NaN to drop using dropna
train['clean_tweet'] = train['clean_tweet'].replace('', np.NaN)


# In[13]:


train.dropna(axis='rows')


# In[14]:


#https://statisticsglobe.com/drop-rows-blank-values-from-pandas-dataframe-python
train['clean_tweet'] = train['clean_tweet'].replace('', float('NaN'), regex = True)


# In[15]:


train.dropna(inplace= True)
train = train.reset_index(drop=True)


# In[16]:


first_column = train.pop('label')
train.insert(0,'label',first_column)


# In[17]:


#total data entries for training

print(train.shape)


# In[18]:


# check for 0 values in train
sum(train["label"] == 0)


# In[19]:


# check for 1 values in train
sum(train["label"] == 1)


# In[20]:


#remove stopwords
import urllib.request, json 
with urllib.request.urlopen("https://raw.githubusercontent.com/stopwords-iso/stopwords-tl/master/stopwords-tl.json") as url:
    stopwords = json.loads(url.read().decode())
    print(stopwords)


# In[21]:


#data['content2'] =data['Content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train['rm_stpwrds'] = train['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords) ]))


# In[22]:


#tokenization 

import nltk
from nltk.tokenize import WhitespaceTokenizer
train['tokenize'] = train['rm_stpwrds'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize) 
train['tokenize'].head()



# In[23]:


from nltk.stem.wordnet import WordNetLemmatizer
def lema_words(text):
  wnl=WordNetLemmatizer()
  return[wnl.lemmatize(w) for w in text]

train['lematize_nltk']=train['tokenize'].apply(lema_words)  
train[['tokenize','lematize_nltk']].sample(5)


# # Proceeding to Training

# In[24]:


VOWELS = "aeiouAEIOU"
CONSONANTS = "bcdfghklmnngpqrstvwyBCDFGHKLMNNGPQRSTVWY"

""" 
	Affixes
"""
PREFIX_SET = [
	'nakikipag', 'pakikipag',
	'pinakama', 'pagpapa',
	'pinagka', 'panganga', 
	'makapag', 'nakapag', 
	'tagapag', 'makipag', 
	'nakipag', 'tigapag',
	'pakiki', 'magpa',
	'napaka', 'pinaka',
	'ipinag', 'pagka', 
	'pinag', 'mapag', 
	'mapa', 'taga', 
	'ipag', 'tiga', 
	'pala', 'pina', 
	'pang', 'naka',
	'nang', 'mang',
	'sing',
	'ipa', 'pam',
	'pan', 'pag',
	'tag', 'mai',
	'mag', 'nam',
	'nag', 'man',
	'may', 'ma',
	'na', 'ni',
	'pa', 'ka',
	'um', 'in',
	'i',
]

INFIX_SET = [
	'um', 'in',
]

SUFFIX_SET = [
	'syon','dor', 
	'ita', 'han', 
	'hin', 'ing', 
	'ang', 'ng', 
	'an', 'in', 
	'g',
]

PERIOD_FLAG = True
PASS_FLAG = False

def check_vowel(substring):
	"""
		Checks if the substring is a vowel.
			letters: substring to be tested
		returns BOOLEAN
	"""

	return all(letter in VOWELS for letter in substring)


def check_consonant(substring):
	"""
		Checks if the letter is a consonant.
			letter: substring to be tested
		returns BOOLEAN
	"""

	return all(letter in CONSONANTS for letter in substring)

def change_letter(token, index, letter):
	"""
		Replaces a letter in a token.
			token: word to be used
			index: index of the letter
			letter: letter used to replace
		returns STRING
	"""
	
	_list = list(token)
	_list[index] = letter

	return ''.join(_list)

def count_vowel(token):
	"""
		Count vowels in a given token.
			token: string to be counted for vowels
		returns INTEGER
	"""

	count = 0

	for tok in token:
		if check_vowel(tok):
			count+=1

	return count


def count_consonant(token):
	"""
		Count consonants in a given token.
			token: string to be counted for consonants
		returns INTEGER
	"""

	count = 0

	for tok in token:
		if check_consonant(tok):
			count+=1

	return count


def check_validation(token):
    with open('stemmer/validation.txt', 'r') as valid:
        data = valid.read().replace('\n', ' ').split(' ')

    return token in data


def clean_repetition(token, REPETITION):
	"""
		Checks token for repetition. (ex. nakakabaliw = nabaliw)
			token: word to be stemmed repetition
		returns STRING
	"""

	if check_validation(token):
		return token

	if len(token) >= 4:
		if check_vowel(token[0]):
			if token[0] == token[1]:
				REPETITION.append(token[0])
				return token[1:]

		elif check_consonant(token[0]) and count_vowel(token) >= 2:
			if token[0: 2] == token[2: 4] and len(token) - 2 >= 4:
				REPETITION.append(token[2:4])
				return token[2:]
			
			elif token[0: 3] == token[3: 6] and len(token) - 3 >= 4:
				REPETITION.append(token[3:6])
				return token[3:]

	return token

def clean_suffix(token, SUFFIX):
    """
    Checks token for suffixes. (ex. bigayan = bigay)
        token: word to be stemmed for suffixes
    returns STRING
    """

    SUF_CANDIDATE = []

    if check_validation(token):
        return token

    for suffix in SUFFIX_SET:
        if len(token) - len(suffix) >= 3 and count_vowel(token[0:len(token) - len(suffix)]) >= 2 and count_consonant(token[0:len(token) - len(suffix)]) >= 1:
            if token[len(token) - len(suffix): len(token)] == suffix:
                if len(suffix) == 2 and not count_consonant(token[0:len(token) - len(suffix)]) >= 1:
                    continue

                if count_vowel(token[0: len(token) - len(suffix)]) >= 2:
                    if suffix == 'ang' and check_consonant(token[-4]) \
                            and token[-4] != 'r' and token[-5] != 'u':
                        continue

                    #print(token[0: len(token) - len(suffix)] + " : " + suffix)

                    if check_validation(token[0: len(token) - len(suffix)]):
                        SUFFIX.append(suffix)
                        return token[0: len(token) - len(suffix)] + 'a' if suffix == 'ita' \
                            else token[0: len(token) - len(suffix)]

                    elif len(SUF_CANDIDATE) == 0:
                        SUF_CANDIDATE.append(suffix)
                        SUF_CANDIDATE.append(token[0: len(token) - len(suffix)])

    if (len(SUF_CANDIDATE) == 2):
        SUFFIX = SUF_CANDIDATE[0]
        return SUF_CANDIDATE[1][0: len(token) - len(SUFFIX)] + 'a' if SUFFIX == 'ita' \
            else SUF_CANDIDATE[1][0: len(token) - len(SUFFIX)]

    return token


def clean_infix(token, INFIX):
	"""
		Checks token for infixes. (ex. bumalik = balik)
			token: word to be stemmed for infixes
		returns STRING
	"""

	if check_validation(token):
		return token

	for infix in INFIX_SET:
		if len(token) - len(infix) >= 3 and count_vowel(token[len(infix):]) >= 2:
			if token[0] == token[4] and token[1: 4] == infix:
				INFIX.append(infix)
				return token[4:]

			elif token[2] == token[4] and token[1: 3] == infix:
				INFIX.append(infix)
				return token[0] + token[3:]

			elif token[1: 3] == infix and check_vowel(token[3]):
				INFIX.append(infix)
				return token[0] + token[3:]

	return token


def clean_prefix(token,	 PREFIX):
	"""
		Checks token for prefixes. (ex. naligo = ligo)
			token: word to be stemmed for prefixes
		returns STRING
	"""

	if check_validation(token):
		return token

	for prefix in PREFIX_SET:
		if len(token) - len(prefix) >= 3 and \
			count_vowel(token[len(prefix):]) >= 2:

			if prefix == ('i') and check_consonant(token[2]):
				continue

			if '-' in token:	
				token = token.split('-')

				if token[0] == prefix and check_vowel(token[1][0]):
					PREFIX.append(prefix)
					return token[1]

				token = '-'.join(token)

			if token[0: len(prefix)] == prefix:
				if count_vowel(token[len(prefix):]) >= 2:
					# if check_vowel(token[len(token) - len(prefix) - 1]):
				# 	continue

					if prefix == 'panganga':
						PREFIX.append(prefix)
						return 'ka' + token[len(prefix):]
					
					PREFIX.append(prefix)
					return token[len(prefix):]

	return token


def clean_duplication(token, DUPLICATE):
	"""
		Checks token for duplication. (ex. araw-araw = araw)
			token: word to be stemmed duplication
		returns STRING
	"""

	if check_validation(token):
		return token

	if '-' in token and token.index('-') != 0 and \
		token.index('-') != len(token) -  1:

		split = token.split('-')

		if all(len(tok) >= 3 for tok in split):
			if split[0] == token[1] or split[0][-1] == 'u' and change_letter(split[0], -1, 'o') == split[1] or \
				split[0][-2] == 'u' and change_letter(split[0], -2, 'o')  == split[1]:
				DUPLICATE.append(split[0])
				return split[0]

			elif split[0] == split[1][0:len(split[0])]:
				DUPLICATE.append(split[1])
				return split[1]

			elif split[0][-2:] == 'ng':
				if split[0][-3] == 'u':
					if split[0][0:-3] + 'o' == split[1]:
						DUPLICATE.append(split[1])
						return split[1]

				if split[0][0:-2] == split[1]:
					DUPLICATE.append(split[1])
					return split[1]

		else:
			return '-'.join(split)
	
	return token


def clean_repetition(token, REPETITION):
	"""
		Checks token for repetition. (ex. nakakabaliw = nabaliw)
			token: word to be stemmed repetition
		returns STRING
	"""

	if check_validation(token):
		return token

	if len(token) >= 4:
		if check_vowel(token[0]):
			if token[0] == token[1]:
				REPETITION.append(token[0])
				return token[1:]

		elif check_consonant(token[0]) and count_vowel(token) >= 2:
			if token[0: 2] == token[2: 4] and len(token) - 2 >= 4:
				REPETITION.append(token[2:4])
				return token[2:]
			
			elif token[0: 3] == token[3: 6] and len(token) - 3 >= 4:
				REPETITION.append(token[3:6])
				return token[3:]

	return token


def clean_stemmed(token, CLEANERS, REPETITION):
		
	if not token:
		return ""
	
	"""
		Checks for left-over affixes and letters.
			token: word to be cleaned for excess affixes/letters
		returns STRING
	"""

	global PERIOD_FLAG
	global PASS_FLAG

	CC_EXP = ['dr', 'gl', 'gr', 'ng', 'kr', 'kl', 'kw', 'ts', 'tr', 'pr', 'pl', 'pw', 'sw', 'sy'] # Consonant + Consonant Exceptions

	if token[-1] == '.' and PASS_FLAG == False:
		PERIOD_FLAG = True

	if not check_vowel(token[-1]) and not check_consonant(token[-1]):
		CLEANERS.append(token[-1])
		token = token[0:-1]

#	if not check_vowel(token[0]) and not check_consonant(token[0]):
#		CLEANERS.append(token[0])
#		token = token[1:]

	if check_validation(token):
		return token

	if len(token) >= 3 and count_vowel(token) >= 2:
		token = clean_repetition(token,	REPETITION)

		if check_consonant(token[-1]) and token[- 2] == 'u':
			CLEANERS.append('u')
			token = change_letter(token, -2, 'o')

		if token[len(token) - 1] == 'u':
			CLEANERS.append('u')
			token = change_letter(token, -1, 'o')

		if token[-1] == 'r':
			CLEANERS.append('r')
			token = change_letter(token, -1, 'd')

		if token[-1] == 'h' and check_vowel(token[-1]):
			CLEANERS.append('h')
			token = token[0:-1]

		# if token[0] == 'i':
		# 	token = token[1:]

		if token[0] == token[1]:
			CLEANERS.append(token[0])
			token = token[1:]

		if (token[0: 2] == 'ka' or token[0: 2] == 'pa') and check_consonant(token[2]) \
			and count_vowel(token) >= 3:
			
			CLEANERS.append(token[0: 2])
			token = token[2:]

		if(token[-3:]) == 'han' and count_vowel(token[0:-3]) == 1:
			CLEANERS.append('han')
			token = token[0:-3] + 'i'

		if(token[-3:]) == 'han' and count_vowel(token[0:-3]) > 1:
			CLEANERS.append('han')
			token = token[0:-3]

		if len(token) >= 2 and count_vowel(token) >= 3:
			if token[-1] == 'h' and check_vowel(token[-2]):
				CLEANERS.append('h')
				token = token[0:-1]

		if len(token) >= 6 and token[0:2] == token[2:4]:
			CLEANERS.append('0:2')
			token = token[2:]

		if any(REP[0] == 'r' for REP in REPETITION):
			CLEANERS.append('r')
			token = change_letter(token, 0, 'd')

		if token[-2:] == 'ng' and token[-3] == 'u':
			CLEANERS.append('u')
			token = change_letter(token, -3, 'o')

		if token[-1] == 'h':
			CLEANERS.append('h')
			token = token[0:-1]

		if any(token[0:2] != CC for CC in CC_EXP) and check_consonant(token[0:2]):
			CLEANERS.append(token[0:2])
			token = token[1:]

	return token


def tg_stemmer(tokens):

    global PERIOD_FLAG
    global PASS_FLAG

    pre_stem     = inf_stem = suf_stem = rep_stem = \
        du1_stem = du2_stem = cle_stem = '-'
    word_info    = {}
    PREFIX     = []
    INFIX      = []
    SUFFIX     = []
    DUPLICATE  = []
    REPETITION = []
    CLEANERS   = []

    word_info['clean'] = '-'
    stemmed_tokens = []

    for token in tokens:
        word_info = {}
        word_info["word"] = token

        if (PERIOD_FLAG == True and token[0].isupper()) or \
                (PERIOD_FLAG == False and token[0].islower()):
            token = token.lower()
            du1_stem = clean_duplication(token, DUPLICATE)
            pre_stem = clean_prefix(du1_stem, PREFIX)
            rep_stem = clean_repetition(pre_stem, REPETITION)
            inf_stem = clean_infix(rep_stem, INFIX)
            rep_stem = clean_repetition(inf_stem, REPETITION)
            suf_stem = clean_suffix(rep_stem, SUFFIX)
            du2_stem = clean_duplication(suf_stem, DUPLICATE)
            cle_stem = clean_stemmed(du2_stem, CLEANERS, REPETITION)
            cle_stem = clean_duplication(cle_stem, DUPLICATE)

            if '-' in cle_stem:
                cle_stem.replace('-', '')

        else:
            PERIOD_FLAG = False
            cle_stem = clean_stemmed(token, CLEANERS, REPETITION)
            word_info["root"]   = token
            word_info["prefix"] = '[]'
            word_info["infix"]  = '[]'
            word_info["suffix"] = '[]'
            word_info["repeat"] = '[]'
            word_info["dupli"]  = '[]'
            word_info["clean"]  = cle_stem

        stemmed_tokens.append(cle_stem)

    return stemmed_tokens


# In[25]:


#ENGLISH STEMMER

from nltk.stem import WordNetLemmatizer

def english_lemmatizer(token):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token)



# In[26]:


nltk.download('averaged_perceptron_tagger')


# In[27]:


lemmatizer = WordNetLemmatizer()

def lemmatize_tagalog_english(tokens):
    lemmatized_tokens = []

    for token in tokens:
        if wordnet.synsets(token):
            pos = nltk.pos_tag([token])[0][1][0].lower()
            pos = {'a': wordnet.ADJ,
                   'n': wordnet.NOUN,
                   'v': wordnet.VERB,
                   'r': wordnet.ADV}.get(pos, wordnet.NOUN)
            lemmatized_token = lemmatizer.lemmatize(token, pos)
        else:
            lemmatized_token = tg_stemmer([token])[0]
        
        lemmatized_tokens.append(lemmatized_token)

    return lemmatized_tokens

train['lemmatize'] = train['tokenize'].apply(lemmatize_tagalog_english)


# In[28]:


train[['label','text','rm_stpwrds','tokenize','lemmatize']].tail(10)


# In[29]:


# Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(train['lemmatize'], train['label'], test_size=0.3, random_state=2)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(map(' '.join, X_train))
X_test = vectorizer.transform(map(' '.join, X_test))


# In[30]:


param_grid = {'C': [10],
'gamma': [0.1]}

svm_model = SVC(kernel='sigmoid')

grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

svm_model = SVC(kernel='sigmoid', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train, y_train)

joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

y_pred = svm_model.predict(X_test)
y_train_pred = svm_model.predict(X_train)

accuracy = metrics.accuracy_score(y_train, y_train_pred)
print("Training Data Accuracy using metrics.accuracy_score():", accuracy)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test Data Accuracy using metrics.accuracy_score():", accuracy)

# precision = metrics.precision_score(y_test, y_pred)
# print("Precision using metrics.precision_score():", precision)

# recall = metrics.recall_score(y_test, y_pred)
# print("Recall using metrics.recall_score():", recall)


# In[31]:


param_grid = {'C': [10],
              'gamma': [0.1]}

svm_model = SVC(kernel='sigmoid')

grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

svm_model = SVC(kernel='sigmoid', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train, y_train)

joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

y_pred = svm_model.predict(X_test)
y_train_pred = svm_model.predict(X_train)

accuracy = metrics.accuracy_score(y_train, y_train_pred)
print("Training Data Accuracy using metrics.accuracy_score():", accuracy)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test Data Accuracy using metrics.accuracy_score():", accuracy)



# In[32]:


precision, recall, thresholds = precision_recall_curve(y_test, svm_model.decision_function(X_test))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


# In[33]:


from sklearn.decomposition import TruncatedSVD
y_train = np.array(y_train)

# create TruncatedSVD object with desired number of components
svd = TruncatedSVD(n_components=2)

# fit and transform the training data
X_train_svd = svd.fit_transform(X_train)

# transform the test data
X_test_svd = svd.transform(X_test)

# reduce dimensionality to 2D using TruncatedSVD
svm_model = SVC(kernel='sigmoid', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train_svd, y_train)

# plot decision boundary
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train_svd, y_train, clf=svm_model, legend=2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Decision Boundary')
plt.show()


# In[34]:


print("confusion_matrix:")
LABEL=['0','1']
conf=confusion_matrix(y_train,y_train_pred)
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_train, y_train_pred),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

print("\nClassification Report of SVM Classifier on Training Data:\n")
print(classification_report(y_train, y_train_pred))


# In[35]:


import string
import re
import urllib.request, json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

# Download Filipino stopwords
with urllib.request.urlopen("https://raw.githubusercontent.com/stopwords-iso/stopwords-tl/master/stopwords-tl.json") as url:
    stopwords_tl = json.loads(url.read().decode())

def preprocess_text(text):
    # Remove unwanted characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    #Tokenize the text
    tokens = word_tokenize(text)
    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stopwords_tl]
    # Lemmatize the tokens
    lemmatized_tokens = []

    for token in tokens:
        if wordnet.synsets(token):
            pos = nltk.pos_tag([token])[0][1][0].lower()
            pos = {'a': wordnet.ADJ,
                   'n': wordnet.NOUN,
                   'v': wordnet.VERB,
                   'r': wordnet.ADV}.get(pos, wordnet.NOUN)
            lemmatized_token = lemmatizer.lemmatize(token, pos)
        else:
            lemmatized_token = tg_stemmer([token])[0]
        
        lemmatized_tokens.append(lemmatized_token)
        
    # Join the tokens back into a string
    text = ' '.join(lemmatized_tokens)
    return text


# Load the trained SVM model
clf = joblib.load('svm_model.joblib')

# Load the vectorizer fitted on the training data
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Get input from user
sentiment = input("Enter a sentence to analyze: ")

# Preprocess the input text
sentiment_processed = preprocess_text(sentiment)

# Vectorize the input text
sentiment_vectorized = vectorizer.transform([sentiment_processed])

# Predict the sentiment using the trained SVM model
prediction = clf.predict(sentiment_vectorized)

print(prediction)
print(sentiment)
# Print the prediction
if prediction == 1:
    print("Negative Statement")
else:
    print("Positive sentiment")


# In[36]:


get_ipython().run_line_magic('scalene', '--help')

