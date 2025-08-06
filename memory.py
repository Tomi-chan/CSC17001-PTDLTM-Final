import numpy as np
import pandas as pd
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.agents.agent_toolkits import create_structured_tool
from typing import List
from langchain_core.tools import StructuredTool
import os 

class MemoryWorker:
    _instance = None
    _initialized = False 

    def __new__(cls, *args, **kwargs): # Singleton design pattern. Only 1 memory is created
        if not cls._instance:
            cls._instance = super(MemoryWorker, cls).__new__(cls)
            # Initialize your stuff here
            # cls._instance.data = []
            # cls._instance.glance = []
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return  # Skip reinitializing

        self.initialized = True
        print(123)
        self.data = []  # Placeholder for the loaded DataFrame
        self.glance = []
        self.tools = [
            StructuredTool.from_function(
                name="Get_data_at_index",
                func=self.get_data_at_idx,
                description="Retrieve the object located at the specified index"
            ),
            StructuredTool.from_function(
                name="Glance_data",
                func=self.glance_data,
                description="Return a list of all glance messages recorded"
            ),
            
        ]
        
    def add_object(self, obj, description: str):
        idx = len(self.data)
        self.data.append(obj)
        self.glance.append({
            "index": idx,
            "description": description
        })
        # print(f"object added, new glance:{self.glance}")

    def get_data_at_idx(self, idx: int):
        """
        Retrieve the DataFrame located at the specified index.

        Args:
            idx (int): The index of the DataFrame in self.data to retrieve.

        Returns:
            pd.DataFrame: The DataFrame at the specified index.
        """
        return self.data[idx]

    def glance_data(self):
        """
        Return a list of all the description of the data stored. This can help to decide which data to retrieve

        Args: None
        Returns:
            list: A list containing all the glance messages.
        """
        tmp = self.glance.copy()
        tmp.reverse()
        return tmp

    # def get_current_data(self, inp: str):
    #     """
    #     Return the most recently object

    #     Args: None
    #     Returns:
    #         object
    #     """
    #     return self.data[-1]

    def clear_data(self):
        pass
    
memoryworker = MemoryWorker()

# memoryworker.glance_data()