#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import re


# In[2]:


ikea_review = pd.read_csv('ikea_malaysia_google_reviews.csv')


# In[3]:


relevant_columns = ['address','categories/0','categories/1','categoryName', 'city','countryCode','isLocalGuide','likesCount','name','postalCode','publishAt','publishedAtDate','reviewerId','reviewerNumberOfReviews','reviewsCount','stars','state','street','text','textTranslated','title']
ikea_reviews = ikea_review[relevant_columns]


# In[4]:


pd.isna(ikea_reviews['text']).sum()


# In[5]:


# Remove missing data
ikea_reviews = ikea_reviews.dropna(subset=['text'])


# In[6]:


pd.isna(ikea_reviews['text']).sum()


# In[7]:


ikea_reviews.to_csv('ikea_reviews_processed.csv', index=False)


# In[8]:


ikea_reviews = pd.read_csv('ikea_reviews_processed.csv')


# In[26]:


from transformers import pipeline

# Language detection
pipe = pipeline(task="text-classification", model="ivanlau/language-detection-fine-tuned-on-xlm-roberta-base")


# In[57]:


#### Read temp results, or create one if there is no result file yet
try:
    temp_result = pd.read_csv("temp_results.csv")
except:
    temp_result = pd.DataFrame(columns=['label','score'])

#### Run the model on each text one at a time.
try:
    count = 0
    for eac in ikea_reviews['text'].loc[len(temp_result):]: #### .loc to Start where you left off
        try:
            eac_res = pipe(eac)
        except:
                print("sentence too long, taking the first 512 characters")
                eac_res = pipe(eac[:512]) 
                #language detection so doesnt matter if it doesnt read the while text of that review
        
        temp_result = pd.concat([temp_result,pd.DataFrame(eac_res)])

        #### Save every 100 text
        if count % 100 == 0:
            print(count)
            temp_result.to_csv("temp_results.csv")

        count +=1

#### Save if hits error
except:
    print("Hit an error. Stopped.")
    temp_result.to_csv("temp_results.csv")


# In[58]:


language = pd.read_csv('temp_results.csv')


# In[59]:


text_df=pd.DataFrame(language)


# In[60]:


ikea_reviews['language']=text_df['label']


# In[61]:


ikea_reviews.to_csv('ikea_reviews_language.csv')  


# In[62]:


ikea_reviews


# In[14]:


import numpy as np
text_list=ikea_reviews['text'][:50].tolist()


# In[15]:


get_ipython().run_cell_magic('time', '', 'pipe(text_list)\n')


# In[11]:


text_df=pd.DataFrame(pipe(text_list))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


pipe("Hari Raya Haji atau Hari Raya Aidil Adha adalah salah satu perayaan terbesar yang dirayakan oleh umat Islam di seluruh dunia selain Hari Raya Aidil Fitri.")


# In[4]:


pipe(["This restaurant is awesome", "This restaurant is awful"])


# In[12]:


pipe("不錯，購物 逛街 找創意")


# In[32]:


import numpy as np
text_list=ikea_review['text'][:3].tolist()


# In[30]:


print(text_list)


# In[33]:


pipe(text_list)


# In[8]:


from transformers import pipeline

pipe = pipeline(task="text-classification",model="ivanlau/language-detection-fine-tuned-on-xlm-roberta-base")
pipe("Hari Raya Haji atau Hari Raya Aidil Adha adalah salah satu perayaan terbesar yang dirayakan oleh umat Islam di seluruh dunia selain Hari Raya Aidil Fitri.")

