# Databricks notebook source
# MAGIC %md
# MAGIC # Filler Model - Gemini Project v1
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries
# MAGIC

# COMMAND ----------

# %pip install numpy==1.23.5
# %pip install tensorflow==2.12.0
# %pip install keras==2.12.0
# %pip install xgboost==1.7.6

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# Load data
# File location and type
file_location = "/FileStore/tables/filler_head_1/filler_data_head_1_for_prediction_small.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
raw_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, StringType, TimestampType

# Get the column names
column_names = raw_df.columns

# Iterate over the column names and cast the columns
for column in column_names:
    column_ = "`{}`".format(column) if '.' in column else column
    if column == 'datetime_nz':
        raw_df = raw_df.withColumn(column, col(column_).cast(TimestampType()))
    elif column == 'MtFillerHeadData.CwPackageName':
        raw_df = raw_df.withColumn(column, col(column_).cast(StringType()))
    else:
        raw_df = raw_df.withColumn(column, col(column_).cast(DoubleType()))

# COMMAND ----------

#Order data by datetime
raw_df = raw_df.sort(col("datetime_nz").asc())

# COMMAND ----------

#Separate df into target_head_y and features_x
y = raw_df.select('target_head_y')
X = raw_df.drop('target_head_y')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove useless features

# COMMAND ----------

#Drop NetWeightHead01 and DateTime columns
X = X.drop('MtFillerHeadData.NetWeightHead01') # repeated column from target_head_y while extracting from historian
X = X.drop('DateTime') # repeated column from datetime_nz while extracting from historian

# COMMAND ----------

# MAGIC %md
# MAGIC # Create New Features

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import lag, col
from pyspark.sql.types import DoubleType

# Define window specification
window = Window.orderBy("datetime_nz")

# Add time difference feature
X = X.withColumn("time_diff", (unix_timestamp("datetime_nz") - lag(unix_timestamp("datetime_nz")).over(window)).cast(DoubleType()))

# COMMAND ----------

from pyspark.sql.functions import when

#Fill first row NULL value ith 0
X = X.withColumn("time_diff", when(X["time_diff"].isNull(), 0).otherwise(X["time_diff"]))

# COMMAND ----------

# Using PySpark's built-in functions to extract specific parts of a datetime
# (i.e., year, month, day, hour, minute, day of the week) from the "datetime_nz" column.
# New columns for these extracted features are being created in the DataFrame X.
from pyspark.sql.functions import year, month, dayofmonth, hour, minute, dayofweek

X = X.withColumn("year", year("datetime_nz"))
X = X.withColumn("month", month("datetime_nz"))
X = X.withColumn("day", dayofmonth("datetime_nz"))
X = X.withColumn("hour", hour("datetime_nz"))
X = X.withColumn("minute", minute("datetime_nz"))
X = X.withColumn("dayofweek", dayofweek("datetime_nz"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## I stopped here converting from pandas (VSCode) to pyspark (Datbricks). -------------------

# COMMAND ----------

X = X.toPandas()
y = y.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## I continue with pandas from where I worked from VSCode. I need to change for pyspark to optimize speed. ---------------------

# COMMAND ----------

# Create moving averages feature
X['moving_avg_3'] = y.rolling(window=3).mean()
X['moving_avg_5'] = y.rolling(window=5).mean()
X['moving_avg_7'] = y.rolling(window=7).mean()

# COMMAND ----------

# To create a new column 'y' which is the 'x' column lagged by 1 period
X['target_head_y_lag_1'] = y.shift(1)
# To create a new column 'y' which is the 'x' column lagged by 2 period
X['target_head_y_lag_2'] = y.shift(2)
# To create a new column 'y' which is the 'x' column lagged by 2 period
X['target_head_y_lag_3'] = y.shift(3)

# COMMAND ----------

# MAGIC %md
# MAGIC # Handling Categorical Data

# COMMAND ----------

# Decide which categorical column to use in our model
for col in X.columns:
    if X[col].dtype == 'object':
        unique_cat = len(X[col].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col, unique_cat=unique_cat))

# COMMAND ----------

# Check if low frequency values.
print(X['MtFillerHeadData.CwPackageName'].value_counts().sort_values(ascending=False).head(10))

# COMMAND ----------

# Create bucket with low frequency values as 'Others'
X['MtFillerHeadData.CwPackageName'] = ['Others' if x not in ['900gSW', '900g'] else x for x in X['MtFillerHeadData.CwPackageName']]
print(X['MtFillerHeadData.CwPackageName'].value_counts().sort_values(ascending=False).head(10))

# COMMAND ----------

# Create dummy columns for  categorical columns
todummy_list = ['MtFillerHeadData.CwPackageName']

# COMMAND ----------

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df

# COMMAND ----------

X = dummy_df(X, todummy_list)

# COMMAND ----------

# MAGIC %md
# MAGIC # Removing the Datetime

# COMMAND ----------

# Remove datetime_nz column
X.drop('datetime_nz', axis=1, inplace=True)
X.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Handling Missing Data

# COMMAND ----------

#check if any missing data in dataframe
X.isnull().sum().sort_values(ascending=False).head(len(X.columns))

# COMMAND ----------

# Separate numeric, non-numeric and boolean columns
numeric_cols = X.select_dtypes(include=np.number).columns
non_numeric_cols = X.select_dtypes(include='object').columns
boolean_cols = X.select_dtypes(include='boolean').columns

# Initialize imputers
imputer_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_non_numeric = SimpleImputer(missing_values=None, strategy='most_frequent')

# Impute numeric columns with mean
if not numeric_cols.empty:
    X[numeric_cols] = X[numeric_cols].interpolate()
    X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])

# Impute non-numeric columns with most frequent value
if not non_numeric_cols.empty:
    X[non_numeric_cols] = imputer_non_numeric.fit_transform(X[non_numeric_cols])

# Convert boolean columns to integers, impute with most frequent, then convert back to boolean
if not boolean_cols.empty:
    X[boolean_cols] = X[boolean_cols].astype('Int64')  # Use 'Int64' (capital 'I') to preserve NaN
    X[boolean_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[boolean_cols])
    X[boolean_cols] = X[boolean_cols].astype('boolean')


# COMMAND ----------

#check if any missing data in dataframe
X.isnull().sum().sort_values(ascending=False).head(len(X.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use miceforest

# COMMAND ----------

# !pip install git+https://github.com/AnotherSamWilson/miceforest.git
# import miceforest as mf
# # Create kernel. 
# kds = mf.ImputationKernel(
#   df_train,
#   save_all_iterations=True,
#   random_state=100
# )

# # Run the MICE algorithm for 2 iterations
# kds.mice(2)

# # Return the completed dataset.
# df_imputed = kds.complete_data()

# COMMAND ----------

# MAGIC %md
# MAGIC # Outliers Detection

# COMMAND ----------

y.count()

# COMMAND ----------

# Remove outliers from dataframe based on df['target_head_y'] with sigma = 3
def remove_outliers(df, column, sigma=3):
    return df[np.abs(df[column] - df[column].mean()) <= (sigma * df[column].std())]

df = pd.concat([X, y], axis=1)
df = remove_outliers(df, 'target_head_y', sigma=3)
y = df['target_head_y']
X = df.drop('target_head_y', axis=1)



# COMMAND ----------

#Use pyplot in matplotlib to plot histogram of df['target_head_y']
%matplotlib inline

def plot_histogram(df, bins=2000):
    plt.figure(figsize=(12, 6))
    plt.hist(df, bins=bins)
    plt.title('Histogram of Target Head Y')
    plt.xlabel('Target Head Y')
    plt.ylabel('Frequency')
    plt.xlim(885, 950)
    plt.show()

plot_histogram(y)

# COMMAND ----------

y.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dimensionality Increase (Not creating Them for the moment)

# COMMAND ----------

# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions for all features
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    # Remove interaction terms with all 0 values
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)

    return df

# X = add_interactions(X)

# COMMAND ----------

# Concatenate the original dataframe and the dataframe with interactions, without consider the index
# df = pd.concat([df.iloc[:, :2].reset_index(drop=True), df_poly], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save DF

# COMMAND ----------

# Keep a copy of your dataframe
X_df = X.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC # Use scaler

# COMMAND ----------

# Use Scaler to scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(X)
X = scaler_X.transform(X)

# COMMAND ----------

# Use Scaler to scale the data
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(y.values.reshape(-1, 1))
y = scaler_y.transform(y.values.reshape(-1, 1))

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection with Boruta or # !pip install BorutaShap

# COMMAND ----------

from boruta import BorutaPy
# Initialize a random forest estimator. Ensure that it's set to handle the nature of your prediction task (classification or regression)
forest = RandomForestRegressor(n_jobs=-1, max_depth=5)

# Initialize Boruta
boruta = BorutaPy(estimator=forest, n_estimators='auto', verbose=2, random_state=1, max_iter=100)

# Fit Boruta (it accepts numpy arrays, not pandas dataframes)
boruta.fit(np.array(X), np.array(y))

# COMMAND ----------

# Print the top features selected by Boruta
top_features = X_df.columns[boruta.support_].to_list()
print("Top features: ", top_features)

# To get features ranking
features_ranking = [X_df.columns[i] for i in boruta.ranking_ - 1]
print("Features ranking: ", features_ranking)


# COMMAND ----------

import seaborn as sns
# Create a pandas DataFrame to hold the feature names and their rankings
ranking_df = pd.DataFrame({'Feature': features_ranking, 'Ranking': boruta.ranking_})

# Sort the DataFrame by the rankings
ranking_df.sort_values(by='Ranking', inplace=True)

# Plot the sorted DataFrame
plt.figure(figsize=(12, 6))
sns.stripplot(x=ranking_df['Ranking'], y=ranking_df['Feature'], orient='h', size=10)
plt.title("Boruta's Rankings of the Features")
plt.xlabel('Ranking')
plt.ylabel('Features')
plt.show()



# COMMAND ----------

import matplotlib.pyplot as plt

# Get the feature importances from Boruta
feature_importances = boruta.feature_importances_

# Create a pandas DataFrame to hold the feature names and their importances
importances_df = pd.DataFrame({'Feature': X_df.columns, 'Importance': feature_importances})

# Sort the DataFrame by the importances
importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot the importances
plt.figure(figsize=(10, 6))
sns.boxplot(data=importances_df, y='Feature', x='Importance', orient='h', color='royalblue')
plt.title('Feature Importances according to Boruta')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection ad Model Building

# COMMAND ----------

# Split dataframe into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# COMMAND ----------

import pandas as pd
from scipy.stats import pearsonr

# COMMAND ----------

# correlations = {}
# for feature in X:
#     corr_test = pearsonr(X, y)
#     correlations[feature] = {"correlation_coefficient": corr_test[0], "p-value": corr_test[1]}

# # Now, correlations dictionary contains correlation coefficients and p-values for each feature.
# # You can convert this dictionary into a DataFrame for a better visualization:

# correlations_df = pd.DataFrame.from_dict(correlations, orient='index')

# COMMAND ----------

import dask.dataframe as dd

# Assume df is your large DataFrame
ddf = dd.from_pandas(X, npartitions=10)  # Increase npartitions based on your system's memory
correlations = ddf.corr().compute()

# COMMAND ----------

# import seaborn as sns
# # Plot correlations in a a heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlations, annot=True, cmap='coolwarm')
# plt.show()

# COMMAND ----------

# corr_df = pd.Dataframe('r', 'p-value')

# for col in df:
#     if pd.api.types.is_numeric_dtype(df[col]) and col != 'target_head_y':
#         r, p = pearsonr(df['target_head_y'], df[col])
#         corr_df[col] = pearsonr(df[col], df['target_head_y'])

# COMMAND ----------

# Remove features with correlation coefficient less than 0.1
def remove_features(df, threshold):
    return df[[column for column in df if abs(df[column].iloc[-1]) > threshold]]
corr_df = remove_features(correlations, 0.5)

# COMMAND ----------

# Select upper triangle of correlation matrix
upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation less than 0.1
to_drop = [column for column in upper.columns if any(upper[column] < 0.1)]

# Drop the features 
df.drop(to_drop, axis=1, inplace=True)

# COMMAND ----------

correlations

# COMMAND ----------

df.columns
