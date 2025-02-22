import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = "Group 16 - Seattle - Homepage.csv"

review_columns_names = ["Restaurant", "Link", "Reviewer", "Rating", "Date", "Number_Reviews", "State", "Review"]
reviews_file_path = "Group 16 - Seattle - review missing restaurants compleet.csv"
reviews_df = pd.read_csv(reviews_file_path, header=None, names=review_columns_names)

df = pd.read_csv(file_path)
print(df.head())


def convert_to_number(value):
    if isinstance(value, str) and 'k' in value:
        return float(value.replace('k', '')) * 1000
    else:
        return float(value)

### Clean Unhelpful Reviews
value_counts = reviews_df.iloc[:, 0].value_counts()
review_counts = reviews_df["Restaurant"].value_counts().reset_index()
review_counts.columns = ["Name", "Unhelpful Reviews"]

df = df.merge(review_counts, on="Name", how="left")

### Clean Restaurant Data
df["Reviews"] = df["Reviews"].str[1:-8]
df["Reviews"] = df["Reviews"].apply(convert_to_number)

df= df.dropna()
print(df.head())


### Regression on number of reviews and review score
# Define independent (X) and dependent (y) variables
X = df[["Reviews"]]  
X_log = np.log1p(X)  
y = df["Rating"]  

# Add a constant for the intercept term
X_log = sm.add_constant(X_log)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Train the model using statsmodels
model = sm.OLS(y_train, X_train).fit()

# Display summary
print(model.summary())

### Regression on number of unhelpful reviews and review score
# Define independent (X) and dependent (y) variables
X = df[["Unhelpful Reviews"]]  
X_log = np.log1p(X)  
y = df["Rating"]  

# Add a constant for the intercept term
X_log = sm.add_constant(X_log)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Train the model using statsmodels
model = sm.OLS(y_train, X_train).fit()

# Display summary
print(model.summary())
