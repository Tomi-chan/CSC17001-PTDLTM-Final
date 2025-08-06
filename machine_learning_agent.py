import pandas as pd
from langchain.tools import StructuredTool
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
from auto_save import auto_save_result
from memory import memoryworker
from sklearn.model_selection import train_test_split


class PredictAgent:
    def __init__(self, memoryworker):
        self.memory = memoryworker
        self.tools = [
            StructuredTool.from_function(
                name="Run_KMeans",
                func=self.kmeans_cluster,
                description="Apply KMeans clustering to selected numeric columns. You should use convert_datetime_data_to_numeric before doing this"
            ),
            StructuredTool.from_function(
                name="Run_LinearRegression",
                func=self.linear_regression,
                description="Train linear regression with specified feature and target columns. You should use convert_datetime_data_to_numeric before doing this"
            ),
            StructuredTool.from_function(
                name="Run_ARIMA",
                func=self.run_arima,
                description="Apply ARIMA to univariate time series column. You should use convert_datetime_data_to_numeric before doing this"
            ),
            StructuredTool.from_function(
                name="convert_datetime_data_to_numeric",
                func=self.convert_datetime_to_numeric,
                description="Convert datetime data into numeric data"
            ),
            StructuredTool.from_function(
                name="Evaluate_MSE",
                func=self.evaluate_mse,
                description="Calculate Mean Squared Error between two numerical columns"
            ),
        ]

    @auto_save_result(memoryworker)
    def evaluate_mse(self, idx: int, y_true_col: str, y_pred_col: str):
        df = self.memory.get_data_at_idx(idx)
        mse = mean_squared_error(df[y_true_col], df[y_pred_col])
        return mse, f"MSE between '{y_true_col}' and '{y_pred_col}' is {mse:.4f}"

    @auto_save_result(memoryworker)
    def kmeans_cluster(
        self,
        idx: int,
        columns: list,
        n_clusters: int = 3
    ):
        df = self.memory.get_data_at_idx(idx)
        X = df[columns]
        model = KMeans(n_clusters=n_clusters, random_state=0)
        df['cluster'] = model.fit_predict(X)
        return df, f"KMeans with {n_clusters} clusters applied on columns {columns} for dataframe at {idx}"

    @auto_save_result(memoryworker)
    def linear_regression(
        self,
        idx: int,
        feature_cols: list,
        target_col: str
    ):
        df = self.memory.get_data_at_idx(idx)
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_test = model.predict(X_test)
        return df, f"Linear regression with features {feature_cols} and target {target_col} for dataframe at {idx}"

    @auto_save_result(memoryworker)
    def run_arima(
        self,
        idx: int,
        column: str,
        order: tuple = (1, 1, 1),
        steps: int = 10
    ):
        df = self.memory.get_data_at_idx(idx)
        series = df[column]
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        result_df = pd.DataFrame({'forecast': forecast})
        return result_df, f"ARIMA({order}) forecast for column '{column}' with {steps} steps for dataframe at {idx}"
    
    @auto_save_result(memoryworker)
    def convert_datetime_to_numeric(self,idx:int):
        df = self.memory.get_data_at_idx(idx)
        df['timestamp'] = df['datetime'].astype('int64') // 1e9  # in seconds
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
        return df,f"Convert datetime data in dataframe {idx} into numeric for dataframe at {idx}"
