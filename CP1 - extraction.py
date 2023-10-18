#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium.webdriver.common.by import By
import pandas as pd
import time


# In[18]:


from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


#driver = webdriver.Chrome()
driver = webdriver.Firefox()

#IKEA Damansara URL
#url = 'https://www.google.com/maps/place/IKEA+Damansara/@3.1574832,101.613178,17z/data=!4m8!3m7!1s0x31cc4f92c5aaca75:0x23ff9ad0a4342afc!8m2!3d3.1574832!4d101.613178!9m1!1b1!16s%2Fg%2F11bx1yv497'

#IKEA Batu Kawan URL
url = 'https://www.google.com/maps/place/IKEA+Batu+Kawan/@5.2332172,100.4386354,17z/data=!4m8!3m7!1s0x304ab7f8c8d8febd:0x9b5254e29e148f89!8m2!3d5.2332172!4d100.4408241!9m1!1b1!16s%2Fg%2F11hyww6f7d?entry=ttu'


driver.get(url)

#time.sleep(5)


# In[3]:


driver.title


# In[4]:


ele = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]')
# ele = driver.find_element(By.XPATH,'//*[@id="pane"]/div/div[1]/div/div/div[2]')

driver.execute_script('arguments[0].scrollBy(200,10000);', ele)
# driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', 
#                 ele)

# driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")


# In[19]:


from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


actions = ActionChains(driver)
for _ in range(1000):
    actions.send_keys(Keys.SPACE).perform()    
    time.sleep(3)


# In[20]:


count = 0
for i in driver.find_elements(By.CSS_SELECTOR,"button"):
    if i.text == "More":
        i.click()
        time.sleep(1.5)
        count +=1
        time.sleep(3)


# In[10]:


name_list = []
stars_list = []
review_list = []
duration_list = []


# In[23]:


get_ipython().run_cell_magic('time', '', 'name_list = []\nstars_list = []\nreview_list = []\nduration_list = []\n\n\nname = driver.find_elements(By.CLASS_NAME,"d4r55")\nstars = driver.find_elements(By.CLASS_NAME,"kvMYJc")\nreview = driver.find_elements(By.CLASS_NAME,"wiI7pd")\nduration = driver.find_elements(By.CLASS_NAME,"rsqaWe")\n\nfor j,k,l,p in zip(name,stars,review,duration):\n    name_list.append(j.text)\n    stars_list.append(k.get_attribute("aria-label"))\n    review_list.append(l.text)\n    duration_list.append(p.text)\n')


# In[24]:


len(name)
# driver.find_elements(By.CLASS_NAME,"d4r55")
#name_list[-1]
#review_list[-1]


# In[25]:


count



# In[26]:


reviews = pd.DataFrame(
    {'name': name_list,
     'rating': stars_list,
     'review': review_list,
     'duration': duration_list})


# In[30]:


reviews.to_csv('ikea_batukawan_google.csv',index=False)
print(reviews)


# In[57]:


review = pd.DataFrame(reviews)

