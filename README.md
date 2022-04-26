# Comparison-of-word-vectors-across-Indian-Languages

We have analysed word embeddings (FasteText,CBOW,SkipGram) of different indian languages(Hindi,Bengali,Gujarati,Marathi,Tamil,Telugu,Kannada).

We have also used language features of the above languages from WALS.Info for our analysis.


To run word embeddings:

1) download the word embeddings for the specified languages from https://www.cfilt.iitb.ac.in/~diptesh/embeddings/monolingual/non-contextual/ and unzip them - word embeddings of dimensions 50 and 100 are used. 
2) Run embedding_cos_sim.ipynb file.


To run WALS features:
1) Download and place Indian_language.csv and language.csv
2) Run the below python files to run for each feature category
  wals_all_features_intersection.py
  wals_feature_pop_lan_repl0.ipynb
  wals_feature_phonology.ipynb
  wals_feature_morphology.ipynb
  wals_feature_nominal.ipynb
  wals_feature_word_order.ipynb
  wals_feature_verbal_categories.ipynb




Folders: 
cos_sim : contains cosine similarity outputs of word embeddings for 50 and 100 dimensions.
train   : 50 randomly picked most frequent words translated in each language that are used for learning the transformation.
test    : categories of words translated in each language used for testing and analysis.
plots   : word wise plots for each model across 50 and 100 dimensions for each language with category code as identifier
outputs : outputs and plots for WALS feature based clustering.  

Files:
en_map.txt - shows the words separated by the category codes
Indian_languages.csv - lists the Indian languages and their properties.
language.csv - lists all the 192 features for all the indian languages.
