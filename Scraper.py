################STEP 1: Motherlist (condensed)

# Load packages and list of restaurants
import pandas as pd
from bs4 import BeautifulSoup
import requests
df_restaurants = pd.read_csv("Group 16 - Seattle.csv")

# Add a new column 'not_recommended_link' based on the existing 'url' column
df_restaurants['not_recommended_url'] = df_restaurants['url'].apply(
    lambda x: x.replace('/biz/', '/not_recommended_reviews/') + '?not_recommended_start=0' if '/biz/' in x else x)


################STEP 2: Going through the subpages

#collect variables for restaurants
url = df_restaurants['not_recommended_url'][2]
name_business = df_restaurants['name'][2]


html = requests.get(url)
soup = BeautifulSoup(html.content, 'lxml')

#soup_username = soup.select()
soup_username = soup.select('.review-list-wide .user-display-name') 
soup_username[1:5]

username = []

for name in soup_username:
    username.append(name.string)
    
username[0:5]

# Get ratings
soup_stars=soup.select('.review-list-wide .rating-large') 
soup_stars[0:5]

rating = []

for stars in soup_stars:
    rating.append(stars.attrs['title']) 

rating[0:5]   

# Get rid of text "star rating"
import re
rating  = [re.sub(' star rating', '',  r) for r in rating]

#convert from string to number
rating = [float(i) for i in rating]

#Get date of rating
soup_date=soup.select('.review-list-wide .rating-qualifier') 
soup_date[0:5]

date_review = []

for date in soup_date:
    date_review.append(date.text.strip())

date_review[0:5]

# Get rid of text "Updated review", "Previous review", "\n", and multiple spaces
date_review  = [re.sub('Updated review', '',  dr) for dr in date_review]
date_review  = [re.sub('Previous review', '',  dr) for dr in date_review]
date_review  = [re.sub('\n', '',  dr) for dr in date_review]
date_review  = [re.sub(' ', '',  dr) for dr in date_review]

#Get the review text
html_texts=soup.select('.review-list-wide p') 
html_texts[0:10]

html_text = []

for t in html_texts:
    html_text.append(t.get_text())

html_text

# Remove certain characters
html_text  = [re.sub('\xa0', '',  ht) for ht in html_text]

##Combine everything to dataset
#add name business
name_business_mult = [name_business] * len(username)
url_restaurant_mult = [url] * len(username)

RatingDataSet = list(zip(name_business_mult, url_restaurant_mult, username, rating, date_review, html_text))

df_rating = pd.DataFrame(data = RatingDataSet, columns=['name','url', 'username', 'rating', 'date_review', 'text'])
df_rating.iloc[:10]

df_rating.to_csv('Group 16 - Seattle - first restaurant.csv',index=False,header=True,encoding='utf8')

#####Step 3: Putting Step 2 in a loop

for u in range(0,4):
    try:                                                      
        #collect variables for restaurants
        url = df_restaurants['not_recommended_url'][u]
        name_business = df_restaurants['name'][u]
        
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'lxml')
        
        soup_username = soup.select('.review-list-wide .user-display-name')
        
        username = []
        
        for name in soup_username:
            username.append(name.string)
            
        
        # Get ratings
        soup_stars=soup.select('.review-list-wide .rating-large')
        
        rating = []
        
        for stars in soup_stars:
            rating.append(stars.attrs['title'])
         
        
        # Get rid of text "star rating"
        import re
        rating  = [re.sub(' star rating', '',  r) for r in rating]
        
        #convert from string to number
        rating = [float(i) for i in rating]
        
        #Get date of rating
        soup_date=soup.select('.review-list-wide .rating-qualifier') 
        
        date_review = []
        
        for date in soup_date:
            date_review.append(date.text.strip())
        
        
        # Get rid of text "Updated review", "Previous review", "\n", and multiple spaces
        date_review  = [re.sub('Updated review', '',  dr) for dr in date_review]
        date_review  = [re.sub('Previous review', '',  dr) for dr in date_review]
        date_review  = [re.sub('\n', '',  dr) for dr in date_review]
        date_review  = [re.sub(' ', '',  dr) for dr in date_review]
        
        #Get the review text
        html_texts=soup.select('.review-list-wide p')
        
        html_text = []
        
        for t in html_texts:
            html_text.append(t.get_text())
        
        html_text
        
        # Remove certain characters
        html_text  = [re.sub('\xa0', '',  ht) for ht in html_text]

        
        name_business_mult = [name_business] * len(username)
        url_restaurant_mult = [url] * len(username)

        RatingDataSet = list(zip(name_business_mult, url_restaurant_mult, username, rating, date_review, html_text))

        df_rating = pd.DataFrame(data = RatingDataSet, columns=['name', 'url', 'username', 'rating', 'date_review', 'text'])

        with open('Group 16 - Seattle - reviews 4 restaurants.csv', 'a',newline='') as f:
            df_rating.to_csv(f, index=False, header=False, encoding='utf8')
                
        print(u)
        import time
        time.sleep(2)
        
    except:
        print("A page was not loaded correctly")

