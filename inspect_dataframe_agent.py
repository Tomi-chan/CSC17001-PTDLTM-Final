from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import List
import os
from langchain_core.tools import StructuredTool
import importlib

# importlib.reload(module= auto_save)
# importlib.reload(module=memory)

from auto_save import auto_save_result
from memory import memoryworker

class InspectAgent:
    def __init__(self, memoryworker):
        self.tools = [
            StructuredTool.from_function(
                name="Load_dataFrame",
                func= self.load_data,
                description="Load a CSV file at filepath into memory"
            ),
            StructuredTool.from_function(
                name="Get_columns",
                func= self.get_column_info,
                description="Retrieve columns' information from the DataFrame given the specified index"
            ),
            StructuredTool.from_function(
                name= "Select_columns",
                func= self.select_columns,
                description="Select and return specific columns from the DataFrame given the specified index"
            ),
            
            StructuredTool.from_function(
                name = "get_head_DataFrame",
                func=self.getHeadDataFrame,
                description= "Return the data of the head of the DataFrame given the specified index"
            ),
            StructuredTool.from_function(
                name = "get_column_names",
                func=self.get_column_names,
                description= f"Return the column name of the dataframe at index specified"
            ),
            StructuredTool.from_function(
                name = "Describe_DataFrame",
                func= self.describe_data,
                description= "Return statistical properties for the dataframe given the specified index"
            ),
            StructuredTool.from_function(
                name= "filter",
                func= self.filter,
                description= "Return a dataframe of entities those satisfy a condition"
            ),
            StructuredTool.from_function(
                name= "merge_dataframe",
                func= self.merge_dataframe,
                description= "Merge two dataframe, these two dataframe should has at least 1 column in common "
            ),
            StructuredTool.from_function(
                name= "rename_column",
                func= self.rename_column,
                description= "Rename the column of the dataframe. For example, \
                    after filter using company name and select_column 'Date' and 'Close Price', \
                        you should change the column name 'Close Price' into 'Brand_Name' "
            )
        ]
        self.memory = memoryworker

    @auto_save_result(memoryworker)
    def load_data(self, filepath: str) -> None:
        """
        Load a CSV file into dataframe and store it into 

        Args:
            filepath (str): The path to the CSV file to be loaded.
        """
        filepath = filepath.strip('`').strip()
        df = pd.read_csv(filepath)
        namefile = os.path.basename(filepath)
        return df,f"Loaded a new dataframe from: {namefile}"

    def get_column_info(self, idx:int ) -> list:
        """
        Get a list of all column names.

        Args:
            idx (int): The index of the DataFrame in self.data from which to retrieve column names.
        """
        return self.memory.get_data_at_idx(idx).info()

    def get_column_names(self, idx:int):
        return self.memory.get_data_at_idx(idx).columns.tolist()

    @auto_save_result(memoryworker)
    def select_columns(self, columns: list[str], idx: int) -> pd.DataFrame:
        """
        Select and return specific columns.

        Args:
            columns (list): A list of column names to select from the DataFrame.
            idx (int): The index of the DataFrame in self.data from which to select columns.
        """
        res = self.memory.get_data_at_idx(idx)[columns]
        return res, f"Get columns = {columns} of DataFrame at index {idx}"

    @auto_save_result(memoryworker)
    def filter(self, idx:int, att: str, val: object):
        df = self.memory.get_data_at_idx(idx)
        df = df.loc[df[att]==val.lower()]
        # df = df.rename()
        return df, f"Extract entities that have attribute {att} equal to {val}"
   
    @auto_save_result(memoryworker)
    def merge_dataframe(self, idx1:int, idx2: int, att):
        df1 = self.memory.data[idx1].reset_index(drop=True)
        df2 = self.memory.data[idx2].reset_index(drop=True)
        assert len(df1) == len(df2)
        merge_df = pd.concat([df1,df2],axis=1)
        return merge_df, f"Merge dataframe {idx1} with dataframe {idx2}"

    @auto_save_result(memoryworker)
    def rename_column(self,idx:int, olds: list[str], news: list[str]):
        df = self.memory.get_data_at_idx(idx)
        scheme = {old:new for old,new in zip(olds,news)}
        df = df.rename(scheme, axis=1)
        return df, f"Rename the column of the dataframe {idx} following the schema{scheme}"

    def describe_data(self, idx: int):
        """
            Provide summary statistical properties for the dataframe

            Args:
                df: pd.DataFrame
        """
        return self.memory.get_data_at_idx(idx).describe()
    

    def getHeadDataFrame(self, idx:int):
        """
        Return the head of the DataFrame

        Args: 
            df: pd_DataFrame
        """
        return self.memory.get_data_at_idx(idx).head()
    
    
# if __name__ == "__main__":
#     a = InspectAgent()
#     a.load_data("D:\\code\\HCMus\\smart_DA\\data\\World-Stock-Prices-Dataset.csv\n")