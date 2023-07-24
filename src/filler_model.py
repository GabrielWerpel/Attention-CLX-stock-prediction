# Databricks notebook source
# MAGIC %md
# MAGIC # Filler Model - Gemini Project
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries
# MAGIC

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
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# COMMAND ----------

# Load data
raw_df = pd.read_csv(r"C:\Users\WERPELGA\Desktop\Databricks\01.IDE_Github_Databricks_Sync\filler_data_head_1_for_prediction_small.csv")

# COMMAND ----------

#Order data by datetime
raw_df.sort_values(by=['datetime_nz'], inplace=True, ascending=True)

# COMMAND ----------

#Separate df into target_head_y and features_x
y = raw_df['target_head_y']
X = raw_df.drop('target_head_y', axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove useless features

# COMMAND ----------

#Drop NetWeightHead01 and DateTime columns
X.drop('MtFillerHeadData.NetWeightHead01', axis=1, inplace=True) # repeated column from target_head_y while extracting from historian
X.drop('DateTime', axis=1, inplace=True) # repeated column from datetime_nz while extracting from historian

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove useless features

# COMMAND ----------

#Drop NetWeightHead01 and DateTime columns
X.drop('MtFillerHeadData.NetWeightHead01', axis=1, inplace=True) # repeated column from target_head_y while extracting from historian
X.drop('DateTime', axis=1, inplace=True) # repeated column from datetime_nz while extracting from historian

# COMMAND ----------

# MAGIC %md
# MAGIC # Create New Features

# COMMAND ----------

#Change the datetime column to datetime format
X['datetime_nz'] = pd.to_datetime(X['datetime_nz'])

# COMMAND ----------

# Add time difference feature
X['datetime_nz_diff'] = X['datetime_nz'].diff().dt.total_seconds()

# Fill the first row from time_diff with value 0
X['datetime_nz_diff'].iloc[0] = 0


# COMMAND ----------

# Create features for the datetime column
X['datetime_nz_year'] = X['datetime_nz'].dt.year
X['datetime_nz_month'] = X['datetime_nz'].dt.month
X['datetime_nz_day'] = X['datetime_nz'].dt.day
X['datetime_nz_hour'] = X['datetime_nz'].dt.hour
X['datetime_nz_minute'] = X['datetime_nz'].dt.minute
X['datetime_nz_dayofweek'] = X['datetime_nz'].dt.dayofweek

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
# MAGIC ### Dimensionality Increase

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

X_df.head()

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
# MAGIC ### PCA

# COMMAND ----------

#Use PCA from sklearn.decomposition to reduce the dimensionality of the dataframe to 10 principal components
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_pca = pd.DataFrame(pca.fit_transform(scaled_data)
# X = df.iloc[:,2:]
# y = df.iloc[:,1]

# COMMAND ----------

# print(scaled_data.shape)
# print(X_pca.shape)

# COMMAND ----------

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca.iloc[:,0],X_pca.iloc[:,1],c=y,cmap='plasma')
# plt.xlabel('First principal component')
# plt.ylabel('Second Principal Component')

# COMMAND ----------

# df_comp = pd.DataFrame(pca.components_,columns=X.columns)
# plt.figure(figsize=(12,6))
# sns.heatmap(df_comp,cmap='plasma',)

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
