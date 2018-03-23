# SIMPLE INTEGRATION BETWEEN GOOGLE BIGQUERY AND H2O AUTOML
A simple script to perform queries to Google BigQuery and perform analytics using H2O.ai AutoML

Docstrings and notes in if-main statement should explain most of what is happening

# Quick Walkthrough

  1. Instantiate class by calling GoogleH2OIntegration(), pass it two string
     arguments to specify which dataset is to be used and what the name of the
     table for the new predictions will be called

  2. Make a query to Google BigQuery using method bigquery_query(). This will
     return a DataFrame that can be worked on for feature engineering as
     necessary

  3. Perform needed feature engineering

  4. Pass data into H2O as H2OFrames.

  5. Call H2O AutoML to perform analytics and make predictions

  6. Zip predictions into list of tuples with test_ids

  7. Pass resultant list to bigquery client to insert into newly created
     predictions table
