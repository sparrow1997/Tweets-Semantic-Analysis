
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


tweets_data = pd.read_csv('Tweets.csv')


# In[3]:


tweets_data.head()


# In[4]:


mood_count = tweets_data['airline_sentiment'].value_counts()


# In[5]:


mood_count


# In[6]:


tweets_data['airline'].value_counts()


# In[7]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[8]:


Index = [1,2,3]
plt.bar(Index, mood_count)
plt.xticks(Index, ['neg', 'neutral', 'pos'])
plt.ylabel('Sentiment Count')


# In[9]:


df_airline_united = tweets_data[tweets_data['airline'] == 'United']


# In[10]:


tweets_data.head(25)


# In[11]:


from wordcloud import WordCloud, STOPWORDS


# In[12]:


df = tweets_data[tweets_data['airline_sentiment'] == 'negative']


# In[13]:


df.head()


# In[14]:


words = ' '.join(df['text'])


# In[15]:


wordcloud = WordCloud(stopwords = 
                      STOPWORDS, background_color='black', 
                      height = 2500, width = 3000).generate(words)


# In[16]:


plt.imshow(wordcloud)
plt.axis('off')


# In[17]:


tweets_data['sentiment'] = tweets_data['airline_sentiment'].apply(

lambda x: 0 if x =='negative' else 1)


# In[18]:


from nltk.corpus import stopwords


# In[19]:


def tweet_to_words(raw_tweet):

    words = raw_tweet.lower().split()
    stopw = set(stopwords.words("english"))
    
    meaningful_words = [w for w in words if not w in stopw]
    
    return (" ".join(meaningful_words))


# In[20]:


#tweets_data['clean_tweets'] = tweets_data['text'].apply(tweet_to_words)


# In[22]:


data = tweets_data['text']


# In[23]:


target = tweets_data['sentiment']


# In[24]:


from sklearn.cross_validation import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer


# In[37]:


v = CountVectorizer(analyzer = "word", ngram_range = (1,2))


# In[31]:


train_features = v.fit_transform(x_train)


# In[32]:


test_features = v.transform(x_test)


# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


clf = DecisionTreeClassifier()
clf.fit(train_features, y_train)
clf.score(test_features, y_test)


# In[35]:


from sklearn.naive_bayes import MultinomialNB


# In[36]:


clf = MultinomialNB()
clf.fit(train_features, y_train)
clf.score(test_features, y_test)


# In[42]:




