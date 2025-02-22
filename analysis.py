import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats

restaurants_file_path = "Group 16 - Seattle - Homepage.csv"
restaurants_df = pd.read_csv(restaurants_file_path)

review_columns_names = ["Name","URL","Username","Rating","Date","Number of Reviews","State","Text"]
reviews_file_path = "Reviews everything merged (csv).csv"
reviews_df = pd.read_csv(reviews_file_path, header=0, names=review_columns_names)

print(restaurants_df.head())

### Define functions for data cleaning
# Converts k-formatted numbers to integers (e.g. 7.5k -> 7500). Keeps other numbers intact.
def convert_to_number(value):
    if isinstance(value, str) and 'k' in value:
        return float(value.replace('k', '')) * 1000
    else:
        return float(value)
    
# Converts a string to upper. Leaves other values intact
def convert_to_upper(value):
    if isinstance(value, str):
        return value.upper()
    else:
        return value

# Scrape list of all state abbreviations from wikipedia, because some of the abbreviations on yelp are not states.
url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
tables = pd.read_html(url)
df_states = tables[1]

# Get only the abbreviations
df_states = df_states.iloc[:, [1]]  
df_states.columns = [ "Abbreviation"]
valid_states = set(df_states["Abbreviation"])

# Put unrecognized state abbreviations into a separate category
def categorize_state(state):
    if pd.isna(state) or state.strip() == "": 
        return pd.NA 
    state = state.upper()
    if state in valid_states:
        return state
    else: 
        return "Non-US"

### Clean Unhelpful Reviews
reviews_df["State"] = reviews_df["State"].apply(categorize_state)

review_counts = reviews_df["Name"].value_counts().reset_index()
review_counts.columns = ["Name", "Unhelpful Reviews"]

restaurants_df = restaurants_df.merge(review_counts, on="Name", how="left")

### Clean Restaurant Data
restaurants_df["Reviews"] = restaurants_df["Reviews"].str[1:-8]
restaurants_df["Reviews"] = restaurants_df["Reviews"].apply(convert_to_number)
restaurants_df["Price"] = restaurants_df["Price"].apply(lambda x: x.count("$") if isinstance(x, str) else None)
restaurants_df= restaurants_df.dropna()

### Regression on number of reviews and review score
# Define independent (X) and dependent (y) variables
X = restaurants_df[["Reviews"]]  
X_log = np.log1p(X)  
y = restaurants_df["Rating"]  

# Add intercept
X_log = sm.add_constant(X_log)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Train regression
model = sm.OLS(y_train, X_train).fit()

print("Regression: Number of Reviews and Review Score")
print(model.summary())

### Regression on number of unhelpful reviews and review score
# Define independent (X) and dependent (y) variables
X = restaurants_df[["Unhelpful Reviews"]]  
X_log = np.log1p(X)  
y = restaurants_df["Rating"]  

# Add intercept
X_log = sm.add_constant(X_log)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Train regression
model = sm.OLS(y_train, X_train).fit()

print("Regression: Number of Unhelpful Reviews and Review Score")
print(model.summary())


### Make US geo-heatmap of reviewer's states.
# Convert state_counts to DataFrame
state_counts_df = reviews_df["State"].dropna().value_counts().reset_index()
state_counts_df.columns = ["Abbreviation", "Review_Count"]

state_counts_df["Review_Count"] = state_counts_df["Review_Count"].replace(0, np.nan)
state_counts_df["Log_Review_Count"] = np.log10(state_counts_df["Review_Count"])

# Separate Washington (WA) data
wa_count_df = state_counts_df[state_counts_df["Abbreviation"] == "WA"]
state_counts_df = state_counts_df[state_counts_df["Abbreviation"] != "WA"] 

# Define custom tick labels: 1, 10, 100, 1000, etc.
tickvals = np.arange(state_counts_df["Log_Review_Count"].min(), state_counts_df["Log_Review_Count"].max() + 1, 1)
ticktext = [f"{int(10**val):,}" for val in tickvals]  

# Create the choropleth map (excluding WA)
fig = px.choropleth(
    state_counts_df,
    locations="Abbreviation",
    locationmode="USA-states",
    color="Log_Review_Count",
    color_continuous_scale="Oranges",
    scope="usa",
    title="Review Frequency by U.S. State (Log Scale, Excluding WA)",
)

# Customize color bar to show real counts
fig.update_layout(
    coloraxis_colorbar=dict(
        tickvals=tickvals,  
        ticktext=ticktext,  
        title="Number of Reviews"
    )
)

if not wa_count_df.empty:
    fig.add_trace(px.choropleth(
        wa_count_df,
        locations="Abbreviation",
        locationmode="USA-states",
        color_discrete_sequence=["black"], 
        scope="usa"
    ).data[0])

fig.show()


### Make barcharts for WA / non-WA review count and average rating
wa_count = wa_count_df["Review_Count"].sum()
non_wa_count = state_counts_df["Review_Count"].sum()

wa_average_rating = reviews_df[reviews_df["State"] == "WA"]["Rating"].dropna().mean()
non_wa_average_rating = reviews_df[reviews_df["State"] != "WA"]["Rating"].dropna().mean()

# Define categories and values
categories = ["Washington (WA)", "Other States"]
review_counts = [wa_count, non_wa_count]
average_ratings = [wa_average_rating, non_wa_average_rating]

# Bar chart for review counts
plt.figure(figsize=(6, 4))
plt.bar(categories, review_counts, color=["black", "orange"])
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews: Washington (WA) vs. Other States")
plt.show()

# Bar chart for average ratings
plt.figure(figsize=(6, 4))
plt.bar(categories, average_ratings, color=["black", "orange"])
plt.ylabel("Average Rating")
plt.title("Average Rating: Washington (WA) vs. Other States")
plt.ylim(0, 5)
plt.show()


### Compare restaurant score vs non-recommended reviews score.
# Compute average rating for each restaurant from unhelpful reviews. 
restaurant_avg_ratings = reviews_df.groupby("Name")["Rating"].mean().reset_index()
restaurant_avg_ratings.columns = ["Name", "Unhelpful Rating"]

ratings_df = restaurants_df[["Name", "Rating", "Price"]].merge(restaurant_avg_ratings, on="Name", how="left")
avg_scores = ratings_df[["Rating", "Unhelpful Rating"]].mean()

# Define categories and values
categories = ["Rating", "Unhelpful Rating"]
avg_values = avg_scores.tolist() 

# Show average score from normal vs non-helpful reviews
plt.figure(figsize=(6, 4))
plt.bar(categories, avg_values, color=["black", "orange"])
plt.ylabel("Average of Averages")
plt.ylim(0, 5)
plt.title("Comparison of Average Ratings")
plt.show()

### 
avg_ratings_by_price = ratings_df.groupby("Price")[["Rating", "Unhelpful Rating"]].mean().reset_index()

# Define categories and values
categories = avg_ratings_by_price["Price"].astype(str)  # Convert price categories to strings for labeling
ratings = avg_ratings_by_price["Rating"]
unhelpful_ratings = avg_ratings_by_price["Unhelpful Rating"]

# Define bar width and positions
x = np.arange(len(categories))
width = 0.35

# Create the bar chart
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, ratings, width, label="Rating", color="black")
plt.bar(x + width/2, unhelpful_ratings, width, label="Unhelpful Rating", color="orange")

plt.xlabel("Price Category")
plt.ylabel("Average Rating")
plt.title("Comparison of Ratings and Unhelpful Ratings by Price Category")
plt.xticks(x, categories)  
plt.legend()
plt.show()


### Run regression on rating and word length.
# Feature engineering: measure length
reviews_df["Review_Length"] = reviews_df["Text"].apply(lambda x: len(re.findall(r'\b\w+\b', str(x))))
reviews_filtered = reviews_df[reviews_df["Review_Length"] > 0].copy()

# Apply log transformation (base 10)
reviews_filtered["Log_Review_Length"] = np.log10(reviews_filtered["Review_Length"])

# Set panel structure
reviews_filtered = reviews_filtered.reset_index().set_index(["Name", "index"])

# Define independent (X) and dependent (y) variables
X = reviews_filtered[["Log_Review_Length"]].copy()
X["Intercept"] = 1
y = reviews_filtered["Rating"]

# Run Fixed Effects (FE) model
fe_model = PanelOLS(y, X, entity_effects=True).fit()
print("Fixed Effects Model:\n", fe_model.summary)


# Run Random Effects (RE) model
re_model = RandomEffects(y, X).fit()
print("\nRandom Effects Model:\n", re_model.summary)

# Perform Hausman Test to determine whether to use FE or RE
b_fe = fe_model.params.drop("const", errors="ignore")  # FE estimates
b_re = re_model.params.drop("const", errors="ignore")  # RE estimates

# Compute the difference
diff = b_fe - b_re

print(diff)

# Compute covariance matrices
V_fe = fe_model.cov.drop("const", errors="ignore").drop(columns="const", errors="ignore")
V_re = re_model.cov.drop("const", errors="ignore").drop(columns="const", errors="ignore")

# Compute Hausman statistic
V_diff = V_fe - V_re
hausman_stat = diff.T @ np.linalg.inv(V_diff) @ diff
p_value = 1 - stats.chi2.cdf(hausman_stat, len(diff) - 1)

print("\nHausman Test Statistic:", hausman_stat)
print("p-value:", p_value)

# Decision based on Hausman test result
if p_value < 0.05:
    print("Use Fixed Effects (FE) model (RE is inconsistent).")
else:
    print("Use Random Effects (RE) model (more efficient).")
