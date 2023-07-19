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
