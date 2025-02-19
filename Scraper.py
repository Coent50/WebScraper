################STEP 1: Motherlist (condensed)

# Load packages and list of restaurants
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

def home_page_scrap():

    ha
def non_rec_scrap ():

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
    soup_username[0:5]

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

    #scraping number of reviews per reviewer
    soup_nr_reviews=soup.select('.review-list-wide .review-count b') 
    soup_nr_reviews [0:10]

    nr_reviews = []

    #get out the right number
    for review in soup_nr_reviews:
        text = review.get_text(strip=True)
        nr_reviews.append(text)

    #adjust the text into an integer
    nr_reviews_int = [int(x) for x in nr_reviews]

    #scraping state where people live
    soup_state=soup.select('.review-list-wide .responsive-hidden-small b') 
    soup_state [0:10]

    state = []
    #getting rid of the <b> 

    for states in soup_state:
        text = states.get_text(strip=True)
        state.append(text)

    #getting only the 2 character state abbreviation
    state_abr = [s[-2:] for s in state]

    ##Combine everything to dataset
    #add name business
    name_business_mult = [name_business] * len(username)
    url_restaurant_mult = [url] * len(username)

    RatingDataSet = list(zip(name_business_mult, url_restaurant_mult, username, rating, date_review, nr_reviews_int, state_abr,html_text))

    df_rating = pd.DataFrame(data = RatingDataSet, columns=['name','url', 'username', 'rating', 'date_review','number of reviews', 'state (abreviation)', 'text'])
    df_rating.iloc[:10]

    df_rating.to_csv('Group 16 - Seattle - first restaurant.csv',index=False,header=True,encoding='utf8')

    #####Step 3: Putting Step 2 in a loop
    for u in range(0,4):
        base_url = df_restaurants.loc[u, 'url']
        name_business = df_restaurants.loc[u, 'name']
        
        #make first not recommended url
        not_recommended_url = base_url.replace('/biz/', '/not_recommended_reviews/') + '?not_recommended_start=0'
        
        html=requests.get(not_recommended_url)
        soup = BeautifulSoup(html.content, 'lxml')

        #find number of loops necessary on first page
        nr_reviewpages = soup.select_one('.review-list-wide .arrange_unit--fill').get_text(strip=True) 
        if nr_reviewpages is None:
            pages_nr_int = 1
        else:
            text=nr_reviewpages
            parts = text.strip().split()
            pages_nr_str = parts[-1]
            pages_nr_int = int(pages_nr_str)   
        
        for i in range(pages_nr_int):
            try:                                                      
            
                offset = i*10 

                review_page_url= base_url.replace('/biz/', '/not_recommended_reviews/') + f'?not_recommended_start={offset}'

                #usage of current page
                html_page = requests.get(review_page_url)
                soup_page = BeautifulSoup(html_page.content, 'lxml')

                #scrape usernames
                soup_username = soup_page.select('.review-list-wide .user-display-name')
                
                username = []
                
                for name in soup_username:
                    username.append(name.string)
                            
                # Get ratings
                soup_stars=soup_page.select('.review-list-wide .rating-large')
                
                rating = []
                
                for stars in soup_stars:
                    rating.append(stars.attrs['title'])
                
                
                # Get rid of text "star rating"
                import re
                rating  = [re.sub(' star rating', '',  r) for r in rating]
                
                #convert from string to number
                rating = [float(i) for i in rating]
                
                #Get date of rating
                soup_date=soup_page.select('.review-list-wide .rating-qualifier') 
                
                date_review = []
                
                for date in soup_date:
                    date_review.append(date.text.strip())
                
                
                # Get rid of text "Updated review", "Previous review", "\n", and multiple spaces
                date_review  = [re.sub('Updated review', '',  dr) for dr in date_review]
                date_review  = [re.sub('Previous review', '',  dr) for dr in date_review]
                date_review  = [re.sub('\n', '',  dr) for dr in date_review]
                date_review  = [re.sub(' ', '',  dr) for dr in date_review]
                
                #Get the review text
                html_texts=soup_page.select('.review-list-wide p')
                
                html_text = []
                
                for t in html_texts:
                    html_text.append(t.get_text())
                
                html_text
                
                # Remove certain characters
                html_text  = [re.sub('\xa0', '',  ht) for ht in html_text]

                #scraping number of reviews per reviewer
                soup_nr_reviews=soup_page.select('.review-list-wide .review-count b') 
                soup_nr_reviews [0:10]

                nr_reviews = []

                #get out the right number
                for review in soup_nr_reviews:
                    text = review.get_text(strip=True)
                    nr_reviews.append(text)

                #adjust the text into an integer
                nr_reviews_int = [int(x) for x in nr_reviews]

                #scraping state where people live
                soup_state=soup_page.select('.review-list-wide .responsive-hidden-small b') 
                soup_state [0:10]

                state = []
                #getting rid of the <b> 

                for states in soup_state:
                    text = states.get_text(strip=True)
                    state.append(text)

                #getting only the 2 character state abbreviation

                state_abr = [s[-2:] for s in state]

                #combine into a dataset
                name_business_mult = [name_business] * len(username)
                url_restaurant_mult = [review_page_url] * len(username)

                RatingDataSet = list(zip(name_business_mult, url_restaurant_mult, username, rating, date_review, nr_reviews_int, state_abr, html_text))

                df_rating = pd.DataFrame(data = RatingDataSet, columns=['name','url', 'username', 'rating', 'date_review','number of reviews', 'state (abreviation)', 'text'])

                with open('Group 16 - Seattle - reviews 4 restaurants v2.csv', 'a',newline='') as f:
                    df_rating.to_csv(f, index=False, header=False, encoding='utf8')
                        
                print(u)
                time.sleep(2)
            
            except Exception as e:
                print(f"A page was not loaded correctly: {e}")

