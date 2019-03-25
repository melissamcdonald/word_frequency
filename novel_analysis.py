# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:10:31 2017

@author: melissa
"""

# Import necessary modules
import pandas as pd
       
import requests  
from bs4 import BeautifulSoup
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer



#URLs of top 25 most downloaded books on Gutenberg Project
URLS = {'Dracula': 'http://www.gutenberg.org/files/345/345-h/345-h.htm',
    'Tom_Sawyer': 'http://www.gutenberg.org/files/74/74-h/74-h.htm', 
    'Dolls_House': 'http://www.gutenberg.org/files/2542/2542-h/2542-h.htm',
    'Metamorphosis': 'http://www.gutenberg.org/files/5200/5200-h/5200-h.htm',
    'The_Prince': 'http://www.gutenberg.org/files/1232/1232-h/1232-h.htm',
    'Huck_Finn': 'http://www.gutenberg.org/files/76/76-h/76-h.htm',
    'Grimm': 'http://www.gutenberg.org/files/2591/2591-h/2591-h.htm',
    'Beowulf': 'http://www.gutenberg.org/files/16328/16328-h/16328-h.htm',
    'Moby_Dick': 'http://www.gutenberg.org/files/2701/2701-h/2701-h.htm',
    'Leviathan': 'http://www.gutenberg.org/files/3207/3207-h/3207-h.htm',
 #  'Romance_Lust':'http://www.gutenberg.org/cache/epub/30254/pg30254-images.html',
    'War_Peace': 'http://www.gutenberg.org/files/2600/2600-h/2600-h.htm',
    'Ben_Franklin': 'http://www.gutenberg.org/files/20203/20203-h/20203-h.htm',
    'Wendigo': 'http://www.gutenberg.org/files/10897/10897-h/10897-h.htm',
    'Mrs_Rowlandson': 'http://www.gutenberg.org/files/851/851-h/851-h.htm',
    'Earnest': 'http://www.gutenberg.org/files/844/844-h/844-h.htm',
    'Great_Expectations': 'http://www.gutenberg.org/ebooks/1400',
 #   'Kama': 'http://www.gutenberg.org/files/27827/27827-h/27827-h.htm',
    'Peter_Pan': 'http://www.gutenberg.org/files/16/16-h/16-h.htm'
       }


titles = []
novel_texts = []

for key in URLS:
    url=URLS[key]
    r=requests.get(url)
    html = r.text
    soup=BeautifulSoup(html, 'html5lib')
    titles.append(soup.title.string)
    novel_texts.append(soup.get_text())
    
   
titles_array = np.asarray(titles)
novels_array = np.asarray(novel_texts)



# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(novels_array)

# Get the words: words
words = tfidf.get_feature_names()

# Print words
#print(words)



#Truncated SVD grouping of books
# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Fit the pipeline to articles
pipeline.fit(csr_mat)

# Calculate the cluster labels: labels
labels = pipeline.predict(csr_mat)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'book': titles_array})

# Display df sorted by cluster label
print(df.sort_values('label'))




# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(csr_mat)

# Transform the articles: nmf_features
nmf_features = model.transform(csr_mat)

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles_array)

# Print the NMF features
print(df)




#pd.DataFrame(books)



def wordcounts(url):
    """The wordcounts function takes any url and will analyse words counts
    on the text
    """
    #import requests
    import requests

    #create the response object
    r = requests.get(url)


    #extract the html from the response object
    html = r.text

    #import BeautifulSoup
    from bs4 import BeautifulSoup
    

    #make a beautiful soup object from the html
    soup=BeautifulSoup(html, 'html5lib')

    #get title
    title = soup.title.string
    
    #get text from soup
    text = soup.get_text()
    
    #Tokenize
    # Import RegexpTokenizer from nltk.tokenize
    from nltk.tokenize import RegexpTokenizer

    # Create tokenizer
    tokenizer = RegexpTokenizer('\w+')

    # Create tokens
    tokens = tokenizer.tokenize(text)
    
    #remove header       
    end_of_header = tokens.index('EBOOK')
    
    for i in reversed(range(end_of_header +1)):
            del tokens[i]
            
            
    beginning_of_footer = tokens.index('END')
    
    
    while len(tokens) > beginning_of_footer:
        del tokens[beginning_of_footer]
        
    words = []    
        
    for i in tokens:
        words.append(i.lower())
            

    #import nltk
    import nltk

    #if stop words not downloaded
    #nltk.download('stopwords')

    #get English stop words and print some
    sw = nltk.corpus.stopwords.words('english')
    

    words_ns = []

    for word in words:
       if word != 'â': 
           if word not in sw:
               words_ns.append(word)
        
        
    #import libraries for datavis
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    #figures in line and set visualiztion style
#    %matplotlib inline
    sns.set()


    #create a frequency distribution and plot
    freqdist1 = nltk.FreqDist(words_ns)
    freqdist1.plot(50, title = title)



for key in URLS:
    url=URLS[key]
    wordcounts(url)