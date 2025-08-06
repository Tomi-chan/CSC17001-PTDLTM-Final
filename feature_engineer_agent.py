import numpy as np
import pandas as pd
from langchain_community.llms import OpenAI
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.agents.agent_toolkits import create_structured_tool
from typing import List
from langchain_core.tools import StructuredTool
import auto_save
import memory
# import importlib

# importlib.reload(module=auto_save)
# importlib.reload(module=memory)
from memory import memoryworker

from auto_save import auto_save_result


class Feature_engineer_agent:
    def __init__(self, memoryworker):
        self.tools = [
            StructuredTool.from_function(
                name="compute_log_return",
                func=self.compute_log_return,
                description="Compute log returns from the Close price and return a list of values"
            ),
            StructuredTool.from_function(
                name= "compute_rolling_volatility",
                func= self.compute_rolling_volatility,
                description="Compute rolling standard deviation (volatility) over a window return a list of values"
            ),
            StructuredTool.from_function(
                name="compute_moving_average",
                func= self.compute_moving_average,
                description = "Compute simple moving average over a rolling window return a a list of values"
            ),
            StructuredTool.from_function(
                name = "compute_momentum",
                func = self.compute_momentum,
                description = "Compute price momentum as the difference between today's Close and Close n days ago return a list of values"
            ),
            # StructuredTool.from_function(
            #         name="Add_feature",
            #         func=self.add_feature,
            #         description="Add a new feature (column) with specified values to the DataFrame at the specified index"
            # ),
        ]
        self.memory = memoryworker

    def compute_log_return(self, idx: int) -> List:
        """
        Compute log returns from the Close price.
        
        Log returns are additive over time and stabilize variance,
        making them easier for modeling and analysis.
        
        Input:
            df: DataFrame with 'Close' column.
            
        Output:
            df: DataFrame with new 'log_return' column.
        """
        df = self.memory.get_data_at_idx(idx)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        self.add_feature('log_return',df['log_return'].values,idx)
        return df['log_return'].values

    def compute_rolling_volatility(self, idx:int, window: int = 20) -> List:
        """
        Compute rolling standard deviation (volatility) over a window.
        
        Rolling volatility measures the recent variability in prices,
        and higher volatility often signals upcoming big moves.
        
        Input:
            df: DataFrame with 'Close' column.
            window: Number of periods for volatility calculation.
            
        Output:
            df: DataFrame with new 'volatility_{window}' column.
        """
        df = self.memory.get_data_at_idx(idx)
        df[f'volatility_{window}'] = df['Close'].rolling(window=window).std()
        self.add_feature(f'volatility_{window}',df[f'volatility_{window}'].values,idx)
        return df[f'volatility_{window}'].values

    def compute_moving_average(self, idx:int, window: int = 20) -> List:
        """
        Compute simple moving average over a rolling window.
        
        Moving averages help reveal the underlying trend by smoothing
        out short-term fluctuations.
        
        Input:
            df: DataFrame with 'Close' column.
            window: Number of periods for moving average.
            
        Output:
            df: DataFrame with new 'sma_{window}' column.
        """
        df = self.memory.get_data_at_idx(idx)
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        self.add_feature(f'sma_{window}',df[f'sma_{window}'].values,idx)
        return df[f'sma_{window}'].values


    def compute_momentum(self, idx:int, n_days: int = 10) -> List:
        """
        Compute price momentum as the difference between today's Close and Close n days ago.
        
        Momentum is a measure of the speed and strength of a price move,
        commonly used to detect trends early.
        
        Input:
            df: DataFrame with 'Close' column.
            n_days: Number of periods for momentum calculation.
            
        Output:
            df: DataFrame with new 'momentum_{n_days}' column.
        """
        df = self.memory.get_data_at_idx(idx)
        df[f'momentum_{n_days}'] = df['Close'] - df['Close'].shift(n_days)
        self.add_feature(f'momentum_{n_days}',df[f'momentum_{n_days}'].values,idx)
        return df[f'momentum_{n_days}'].values
    
    @auto_save_result(memoryworker)
    def add_feature(self, feature_name: str, values: list[float], idx: int) :
        """
        Add a new feature (column) to the dataset.

        Useful when agent computes a transformation.

        Args:
            feature_name (str): The name of the new feature (column) to be added.
            values (pd.Series): The values to be assigned to the new feature.
            idx (int): The index of the DataFrame in self.data where the feature will be added.
        """
        tmp = self.memory.get_data_at_idx(idx)
        tmp[feature_name] = values
        print(f"Feature '{feature_name}' added to memory.")
        return tmp, f"Add column {feature_name} into Dataframe at index {idx}"

if __name__ == "__main__":
    a = Feature_engineer_agent()
    print(a.tools)

# llm = OpenAI(temperature=0)

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Let LLM decide steps freely
#     verbose=True
# )

# # 4. Run a Query

# result = agent.run("Get the mean and standard deviation of 'column_a'.")
# print("\nFinal result:\n", result)