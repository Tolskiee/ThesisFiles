def lemmatize_tagalog_english(tokens):
  
  import nltk
  from nltk.corpus import wordnet
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer
  from tg_stem import tg_stemmer
  
  lemmatizer = WordNetLemmatizer()
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
          from tg_stem import tg_stemmer
      lemmatized_tokens.append(lemmatized_token)
      
  # Join the tokens back into a string
  text = ' '.join(lemmatized_tokens)
  
  return text