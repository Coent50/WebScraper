# Load packages and list of restaurants
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

def home_page_scrap():

    df_restaurants = pd.read_csv("Group 16 - Seattle.csv")

    data = [] 

    # Loop through all restaurants in the dataset
    for i in range(0,4):  
    
        #collect variables for restaurants
        url = df_restaurants['url'][i]
        name_business = df_restaurants['name'][i]
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'lxml')

        # Get ratings
        soup_stars=soup.select('.arrange-unit-fill__09f24__CUubG.y-css-1n5biw7 .y-css-1jz061g') 
        rating = []

        for stars in soup_stars:
            rating.append(stars.text.strip()) 

        rating[0:5]   

        # Get reviews count
        soup_review = soup.select('.y-css-r8orer .y-css-1x1e1r2')  
        reviews = []
        
        for review in soup_review:
            reviews.append(review.text.strip())  

        # Get price tag
        soup_price = soup.select('.y-css-1tsir1e')
        price_tag = []
        
        for price in soup_price:
            price_tag.append(price.text.strip()) 

         # Get restaurant type 
        soup_type = soup.select('.y-css-1jz061g .y-css-1x1e1r2')  
        restaurant_type = []
        
        for type in soup_type:
            restaurant_type.append(type.text.strip())  

        # Get claimed
        soup_claim = soup.select('.y-css-1t10hx3')  
        claimed = []
        
        for claim in soup_claim:
            claimed.append(claim.text.strip()) 
        
    # Store data in dictionary
        data.append({
            "Name": name_business,
            "URL": url,
            "Rating": rating[0] if rating else "N/A",
            "Reviews": reviews[0] if reviews else "N/A",
            "Price": price_tag[0] if price_tag else "N/A",
            "Type": restaurant_type[0] if restaurant_type else "N/A",
            "Claimed": claimed[0] if claimed else "N/A"
        })
        df_output = pd.DataFrame(data)
        # Save to CSV
        df_output.to_csv("Group 16 - Seattle - HomePage.csv", index=False)

home_page_scrap()

def non_rec_scrap ():

    df_restaurants = pd.read_csv("Group 16 - Seattle.csv")

    # Add a new column 'not_recommended_link' based on the existing 'url' column
    df_restaurants['not_recommended_url'] = df_restaurants['url'].apply(
        lambda x: x.replace('/biz/', '/not_recommended_reviews/') + '?not_recommended_start=0' if '/biz/' in x else x)

    #collect variables for restaurants
    url = df_restaurants['not_recommended_url'][2]
    name_business = df_restaurants['name'][2]

    # Collecting the subpage data through a loop
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

non_rec_scrap ()