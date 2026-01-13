import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
import langid
from bs4 import BeautifulSoup
import html
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from libretranslatepy import LibreTranslateAPI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier



##import original data
##data is already split into Training/Testing, no need to re-split
Rak_train_raw = pd.read_csv('../data/raw/X_train_update.csv', index_col=0)  ##raw X train data
RakY_train_raw = pd.read_csv('../data/raw/Y_train_CVw08PX.csv', index_col=0)  ##raw Y train data
Rak_test_raw = pd.read_csv('../data/raw/X_test_update.csv', index_col=0)  ##raw X test data

#####
##Clean-up strings
##FIXME def func

##Fix problematic strings for text clean-up

Rak_train = Rak_train_raw.copy(deep = True)

## Replace NaN with ''
## (for some reason strings can include numeric NaN values)
Rak_train['description'] = Rak_train['description'].fillna('')

## lower case
Rak_train['designation'] = Rak_train['designation'].str.lower()
Rak_train['description'] = Rak_train['description'].str.lower()

##FIXME simplify syntax
##Unescape HTML
Rak_train['designation'] = Rak_train.apply(lambda row: html.unescape(row['designation']), axis = 1)
Rak_train['description'] = Rak_train.apply(lambda row: html.unescape(row['description']), axis = 1)

## Remove HTML tags
Rak_train['designation'] = Rak_train.apply(lambda row: BeautifulSoup(row['designation'], "html.parser").get_text(separator=" "), axis = 1)
Rak_train['description'] = Rak_train.apply(lambda row: BeautifulSoup(row['description'], "html.parser").get_text(separator=" "), axis = 1)

##FIXME drop empty designation & description rows
##FIXME make fun to clean-up data (to apply to test as well)


##FIXME list of problematic strings to fix
## àªtre

##Regex replacements
repl_dict = {
             ##FIXME nltk.classify.textcat.TextCat().remove_punctuation() 
             r'n°': r' numéro ', 

             ##FIXME not sure how to handle '¿' or '?'
             ##insert space around any non-digit, non-word and non-whitespace with (e.g. '\?' -> ' \? ', 'n°' -> 'n ° ')
             ##except ¿'
             r"[^\d\w\s¿\?'\-]": r' \g<0> ',  

             ##FIXME possibly remove digits after translation
             r"\b\S*[0-9]+\S*\b": '' ##drop all words that contain digits (so drop all digits as well)
            }

Rak_train = Rak_train.replace(to_replace = {'designation': repl_dict,
                                               'description': repl_dict},
                                 regex = True)



##Concat strings
Rak_train['product_txt'] = Rak_train['designation'] + ' . -//- ' + Rak_train['description']

Rak_train['product_txt_len'] = Rak_train['product_txt'].apply(len)


#####
##foreign language handling
##detect phrase language using 'langid' on product_txt

# langid.set_languages(langs=None)
langid.set_languages(langs=['fr', 'en', 'de', 'it', 'es', 'pt'])
Rak_train['lang'] = Rak_train['product_txt'].apply(lambda x: langid.classify(x)[0])


##FIXME def func with a param for rerun={0,1}; if 1 run translate code, if 0 load csv
##translate using libretranslate (self-hosted process)
##NOTE: need to start external process first
# libretranslate --update-models --load-only fr,en,es,de,it,pt
# libretranslate --load-only fr,en,es,de,it,pt

lt = LibreTranslateAPI("http://localhost:5000/")

##re-use detected language
def lt_fun(row):
    if row.name % 10000 == 0: print(row.name)
    # print(row['lang'])
    transl = row['product_txt'] if row['lang'] == 'fr' else lt.translate(
        row['product_txt'], source=row['lang'], target="fr")
    return transl


Rak_train['product_txt_transl'] = Rak_train.apply(lambda row: lt_fun(row), axis=1)

##save translations to csv
Rak_train.to_csv('../data/processed/Rak_train_translations.csv')


#####
##Training Sample Rebalancing
RakX_train = Rak_train[['product_txt_transl']]

##Oversampling (only on training data)
##SMOTEN
smo = SMOTEN(random_state=27732)
RakX_train_sm, Raky_train_sm = smo.fit_resample(RakX_train, RakY_train_raw)


#####
##Tokenization
# Créer un vectorisateur 
##FIXME consider custom tokenizer and max_features
regexp_tokenizer = RegexpTokenizer("[a-zA-ZÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ]{3,}") ##words with at least 3 characaters
vect_tfidf = TfidfVectorizer()

# Mettre à jour la valeur de X_train_tfidf et X_test_tfidf
##FIXME
RakX_train_sm_tfidf = vect_tfidf.fit_transform(RakX_train_sm)





