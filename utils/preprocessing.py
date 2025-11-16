import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime

class TrafficPreprocessor:
    def __init__(self):
        self.scaler = None
        self.scaler_tv = None
        self.columns = None

    def fit(self, df: pd.DataFrame):
        """
        Fit scaler on training data.
        """
        self.scaler = MinMaxScaler().fit(df)
        self.scaler_tv = MinMaxScaler().fit(df[['traffic_volume']])
        self.columns = df.columns
        return self

    def transform(self, df: pd.DataFrame):
        """
        Apply scaling to dataframe.
        """
        scaled = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)
        return scaled

    def inverse_transform(self, df_scaled: pd.DataFrame):
        """
        Convert scaled data back to original values.
        """
        return pd.DataFrame(self.scaler.inverse_transform(df_scaled), index=df_scaled.index, columns=df_scaled.columns)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def add_time_features(df: pd.DataFrame):
        """
        Add sin/cos time features and date components.
        """
        timestamp_s = df.index.map(datetime.datetime.timestamp)
        day = 24*60*60
        year = 365.2425*day

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        
        df['dayofweek'] = df.index.dayofweek
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_hour'] = df.index.hour
        
        return df
