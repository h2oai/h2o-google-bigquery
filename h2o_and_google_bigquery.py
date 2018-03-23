from google.cloud import bigquery, storage
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import h2oai_client
from h2oai_client import Client, ModelParameters
import numpy as np
import pandas as pd
import h2o
import os


class GoogleH2OIntegration(object):

    def __init__(self, dataset, pred_table_name, gcp_auth=None):
        """
        Constructs class object of GoogleH2OIntegration.
        NOTE: if argument pred_table_name points to an already existing table
              in BigQuery, that table will be deleted
        INPUT: dataset (STRING) - name of dataset from Google bigquery
               pred_table_name (STRING) - name for new table in bigquery
        ATTRIBUTES: self.bq_client - Initialized bigquery client
                    self.s_client - Initialized gcs client
                    self.dataset - dataset reference for bigquery API
                    self.col_name/self.col_type/self.mode - lists of strings
                                  necessary for creating 2 column prediction
                                  table schema
                    self.SCHEMA - self._set_schema() is called becomes a list
                                  of type SchemaField
                    self.pred_table_ref - table reference for bigquery API
                    self.pred_table - instance of new bigquery table
        """
        if gcp_auth.split('.')[-1] == 'json':
            self.bq_client = bigquery.Client.from_service_account_json(gcp_auth)
            self.s_client = storage.Client.from_service_account_json(gcp_auth)
        else:
            self.bq_client = bigquery.Client()
            self.s_client = storage.Client()

        self.dataset = self.bq_client.dataset(dataset)
        self.col_name = ['test_id', 'prediction']
        self.col_type = ['INTEGER', 'STRING']
        self.mode = ['required', 'required']
        self.SCHEMA = []
        self._set_schema(self.col_name,
                         self.col_type,
                         self.mode)
        self.pred_table_ref = self.dataset.table(pred_table_name)
        self.pred_table = bigquery.Table(self.pred_table_ref,
                                         schema=self.SCHEMA)
        try:
            self.bq_client.get_table(self.pred_table_ref)
            self.bq_client.delete_table(self.pred_table_ref)
            self.pred_table = self.bq_client.create_table(self.pred_table)
            print ('There was an existing table')
        except:
            self.pred_table = self.bq_client.create_table(self.pred_table)
            print ('There was NO existing table')

    def bigquery_query(self, gcs_bucket, gcs_path="bq_tmp.csv"):
        """
        METHOD: prompts user to enter StandardSQL query as INPUT.
                Will attempt to dump query to csv file in gcs bucket first
                because it is faster to do this. Larger queries will be
                converted directly to a dataframe but this will take much
                longer

        NOTE: Enter first line of query, hit return,
              next line of query, return.
              Once finished press CTRL + D to complete query

        OUTPUT: dataframe of results from bigquery query
        """
        query = self._multiline()
        t_ref = self.dataset.table("tmp_table")
        table = bigquery.Table(t_ref)

        job_config = bigquery.QueryJobConfig()
        job_config.destination = t_ref
        job_config.write_disposition = "WRITE_TRUNCATE"

        query_job = self.bq_client.query(query, job_config=job_config)
        query_job.result()
        file_size = query_job.total_bytes_processed

        if file_size < 900000000:
            dst = 'gs://' + gcs_bucket + '/' + gcs_path
            job = self.bq_client.extract_table(t_ref, dst)
            job.result()
            bucket = self.s_client.get_bucket(gcs_bucket)
            blob = bucket.get_blob(gcs_path)
            blob.download_to_filename('/tmp/' + gcs_path)
            df = pd.read_csv('/tmp/' + gcs_path)
            os.remove('/tmp/' + gcs_path)
        else:
            df = query_job.to_dataframe()

        self.bq_client.delete_table(t_ref)
        return df

    def write_to_table(self, test_ids, predictions):
        """
        Takes test_ids and predictions and adds them to new predictions table
        INPUT: test_ids (LIST of INTEGERS) - foreign keys for SQL JOIN
               predictions (LIST of STRINGS) - current schema set by default
                                               for strings, can be changed.
                                               All predictions from test set
        """
        to_table = zip(test_ids, predictions)
        self.bq_client.insert_rows(self.pred_table, to_table)
        print ("Success")

    def h2o_automl(self, X_train, X_test, target, h2o_args,
                   aml_args, drop_cols=[], classification=True):
        """
        Initializes an instance of H2O and runs H2O AutoML to identify and
        return top model
        INPUT: X_train (DATAFRAME) - training data with target column
               X_test (DATAFRAME) - validation data (can have target column)
               target (STRING) - name of target column
               drop_cols (LIST of STRINGS) - list of all columns to be ignored
                                             in training
               h2o_args (DICT of kwargs) - dictionary containing all desired
                                           arguments for initializing H2O
               aml_args (DICT of kwargs) - dictionary containing all desired
                                           arguments for initializing AutoML
               classification - (BOOL) True will train AutoML as classification
        OUTPUT: aml - trained AutoML object containing best model (aml.leader)
                preds - predictions based on test dataset
        """
        h2o.init(**h2o_args)

        train_col = list(X_train.columns)
        test_col = list(X_test.columns)

        train = h2o.H2OFrame.from_python(X_train, column_names=train_col)
        test = h2o.H2OFrame.from_python(X_test, column_names=test_col)

        x = train.columns
        x.remove(target)
        for col in drop_cols:
            x.remove(col)

        if classification:
            train[target] = train[target].asfactor()
            test[target] = test[target].asfactor()

        aml = H2OAutoML(**aml_args)
        aml.train(x=x, y=target, training_frame=train)
        lb = aml.leaderboard
        print (lb)
        preds = aml.leader.predict(test)

        return aml, preds

    def _multiline(self):
        print ("Enter/Paste your content. 'end_query' to save it.")
        contents = []
        continue_query = True
        while continue_query:
            line = input("")
            if line == 'end_query':
                continue_query = False
                continue
            contents.append(line)
        return " ".join(contents)

    def _set_schema(self, col_name, col_type, mode):
        for i in range(len(col_name)):
            one_col = bigquery.SchemaField(col_name[i],
                                           col_type[i],
                                           mode=mode[i])
            self.SCHEMA.append(one_col)
