
def preprocess_text(text):
  import re
  import urllib.request, json
  from nltk.tokenize import word_tokenize
  with urllib.request.urlopen("https://raw.githubusercontent.com/stopwords-iso/stopwords-tl/master/stopwords-tl.json") as url: stopwords_tl = json.loads(url.read().decode())
  from eng_tag_lemmatizer import lemmatize_tagalog_english
  # Remove unwanted characters and digits
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  #Tokenize the text
  tokens = word_tokenize(text)
  # Convert tokens to lowercase
  tokens = [word.lower() for word in tokens]
  # # Remove stopwords
  tokens = [word for word in tokens if word.lower() not in stopwords_tl]
  # Lemmatize the tokens

  text = lemmatize_tagalog_english(tokens)

  return text