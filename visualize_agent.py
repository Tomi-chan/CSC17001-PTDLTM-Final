import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain.tools import StructuredTool
from auto_save import auto_save_result
from memory import memoryworker
import openai
import base64
from pathlib import Path
import os
from dotenv import load_dotenv
import shutil
import matplotlib.dates as mdates

load_dotenv("key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
plotfolder = "./plot/"

class VisualizationAgent:
    def __init__(self, memoryworker):
        def refresh_plot_folder():
            directory = "./plot/"
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        refresh_plot_folder()
        self.memory = memoryworker  # Reference to MemoryWorker for accessing data
        self.current_plots = []
        self.tools = [
            StructuredTool.from_function(
                name="Plot_histogram",
                func=self.plot_histogram,
                description="Plot a histogram of a column in a DataFrame. Args: idx (int), column_name (str), bins (int)"
            ),
            StructuredTool.from_function(
                name="Plot_general_chart",
                func=self.plot_general_chart,
                description="Choose an appropriate (scatter, line, bar, violin,kde) and plot it unless the user specifiy from a DataFrame. Args: idx (int), x (str), y (str), kind (str). kind can be 'scatter', 'line', or 'bar', etc"
            ),
            StructuredTool.from_function(
                name="Plot_correlation_matrix",
                func=self.plot_correlation_matrix,
                description="Plot a correlation heatmap from a DataFrame. Args: idx (int)"
            ),
            StructuredTool.from_function(
                name="analyze_plot_image",
                func=self.analyze_plot_image,
                description="Retrieve some insight about the plot retrieved in data. Args: index of the plot"
            ),
        ]
    @auto_save_result(memoryworker)
    def plot_histogram(self, idx: int, column_name: str, bins: int = 20) -> str:
        df = self.memory.get_data_at_idx(idx)
        if column_name not in df.columns:
            return f"Column {column_name} not found in DataFrame at index {idx}"
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column_name], bins=bins, kde=True)
        plt.title(f'Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.xticks(rotation=45)
        plt.ylabel('Frequency')
        plt.tight_layout()
        link = plotfolder+f"histogram_{column_name}.png"
        plt.savefig(link)
        # plt.show()
        self.current_plots.append(link)
        return link,f"Histogram of {column_name} of dataframe {idx} plotted and stored."

    @auto_save_result(memoryworker)
    def plot_general_chart(self, idx: int, x: str, y: str, kind: str = "scatter") -> str:
        df = self.memory.get_data_at_idx(idx)

        if x == 'Date':
            df[x] = pd.to_datetime(df[x])

        if x not in df.columns or y not in df.columns:
            return f"Either '{x}' or '{y}' not found in DataFrame at index {idx}."

        plt.figure(figsize=(8, 5))
        
        if kind == "scatter":
            sns.scatterplot(data=df, x=x, y=y)
        elif kind == "line":
            sns.lineplot(data=df, x=x, y=y)
        elif kind == "bar":
            sns.barplot(data=df, x=x, y=y)
        elif kind == "violin":
            sns.violinplot(data=df, x=x, y=y)
        elif kind == "kde":
            sns.kdeplot(data=df, x=x, y=y)
        else:
            return f"Plot type '{kind}' is not supported. Please use 'scatter', 'line', or 'bar'."

        plt.title(f"{kind.capitalize()} plot of {y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        if x=='Date':
              # x == 'Date'

            sns.lineplot(data=df, x=x, y=y)
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
        link = plotfolder+f"{kind}_{x}_{y}.png"
        plt.savefig(link)
        self.current_plots.append(link)
        # plt.show()
        return link, f"{kind.capitalize()} plot of {y} vs {x} for dataframe {idx} created and stored"

    

    @auto_save_result(memoryworker)
    def plot_correlation_matrix(self, idx: int) -> str:
        df = self.memory.get_data_at_idx(idx)
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()

        link = plotfolder+f"correlation_matrix_of_dataframe_{idx}.png"
        plt.savefig(link)
        self.current_plots.append(link)
        # plt.show()
        return link, f"Correlation matrix of dataframe {idx} plotted and stored."
    

    def analyze_plot_image(self,idx:int, question="What insights can you derive from this plot?"):
        def encode_image(image_path):
            return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        
        image_path = self.memory.get_data_at_idx(idx)
        image_base64 = encode_image(image_path)
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant. Describe insights from the provided plot."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    def have_plots(self):
        return len(self.current_plots) != 0
    
    def get_plots(self):
        return self.current_plots
    
    def clear_current_plots(self):
        self.current_plots = []
    
