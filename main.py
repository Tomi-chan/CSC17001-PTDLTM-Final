
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import List
import os
from langchain.memory import ConversationBufferMemory 
from visualize_agent import VisualizationAgent
from pydantic import BaseModel
from flask import Flask, request, send_file, jsonify
from io import BytesIO
from PIL import Image
import importlib
# import memory
# import feature_engineer_agent
# import inspect_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

# importlib.reload(module=memory)
# importlib.reload(module=feature_engineer_agent)
# importlib.reload(module=inspect_dataframe_agent)    

from machine_learning_agent import PredictAgent
from memory import MemoryWorker, memoryworker
from feature_engineer_agent import Feature_engineer_agent
from inspect_dataframe_agent import InspectAgent


load_dotenv('key.env')

# print(os.getenv('OPENAI_API_KEY'))

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.3  # Optional: makes it deterministic
)



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in data analysis assistant.
            - Most important: You must try to reuse the data which have been created by using glance_data. Do not create data that have already been created
            - When you need to access any existing data, use 'glance_data' to review descriptions provided.
            - Before creating new data, you should use glance_data() to check whether the data has been created before.
            - Choose the most relevant description to find the correct data.
            - You must use the available tools (e.g., get_data_at_idx) to retrieve actual dataframes or series.
            - Think step-by-step and explain your choices when selecting or manipulating data.
            - After you get correct information from a tool, try to answer the question directly.
            - Only call a tool again if absolutely necessary.
            - Finalize your answer with "In conclusion:" followed by your explanation.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


class LeaderAgent:
    def __init__(self, memoryworker):
        self.memoryworker = memoryworker
        self.inspector = InspectAgent(self.memoryworker)
        self.feature_agent = Feature_engineer_agent(self.memoryworker)
        self.visualize_agent = VisualizationAgent(memoryworker)
        self.predict_agent = PredictAgent(memoryworker)

        self.tools = self.memoryworker.tools
        self.tools.extend(self.inspector.tools)
        self.tools.extend(self.feature_agent.tools)
        self.tools.extend(self.visualize_agent.tools)
        self.tools.extend(self.predict_agent.tools)

        agent = create_openai_functions_agent(
            llm=llm,
            tools=self.tools,
            prompt= prompt
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = AgentExecutor(
                tools=self.tools,
                memory = memory,
                agent=agent, 
                verbose=True
            )
    def run(self, user_input):
                # Instead of injecting into prompt manually, we add system message to memory if needed
                # Here you inject glance
        glance_info = self.memoryworker.glance_data()
        # print("glance_info: ",glance_info)
        glance_info = ("\n").join([str(a) for a in glance_info])
        print("glance_info: ",glance_info)
        full_prompt = f"""
            You have access to the following previously loaded data:
            {glance_info}

            User Query:
            {user_input}
        """

        result = self.agent.invoke({"input": full_prompt})
        result['images_link'] = self.visualize_agent.get_plots()
        self.visualize_agent.clear_current_plots()
        return result
    def get_new_data(self):
         return self.memoryworker.glance_data()

beo = LeaderAgent(memoryworker)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
  
        response = beo.run(message)
        content = response.get("output", "No output provided.")
        images_link = response.get("images_link", [])
        return jsonify({
            "content": content,
            "images_link": images_link
        })

    except Exception as e:
        return jsonify({
            "content": f"Error processing message: {str(e)}",
            "images_link": []
        }),500
    
@app.route('/data', methods=['POST'])
def update_data():
    try:
        # data = request.json/
        response = beo.get_new_data()
        # print(response)
    except Exception as e:
            print(f"An error occurred: {e}")

    
    return jsonify({"type": "text", "content": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)