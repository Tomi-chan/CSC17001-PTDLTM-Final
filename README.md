# Automated Data Analysis AI Agent

## Introduction

This project is an AI-powered application designed to automate the data mining and analysis workflow. The system is built as a conversational AI assistant, allowing users to perform complex data tasks using natural language. The workflow spans from data loading, cleaning, and feature engineering to visualization and machine learning modeling.

## Key Features

* **Intuitive User Interface**: A Streamlit-based web application provides a chat interface for users to interact with the AI agent.
* **Multi-Agent System**: Complex tasks are broken down into specialized agents:
    * **Inspect Agent**: Loads and explores the data structure.
    * **Feature Engineer Agent**: Creates new features from existing data, particularly for time-series analysis.
    * **Visualization Agent**: Generates various plots (line, bar, scatter, heatmap, etc.) for data visualization.
    * **Machine Learning Agent**: Applies basic machine learning models (KMeans, Linear Regression, ARIMA) for analysis and prediction.
* **Contextual Memory**: A central memory system stores all created data objects and a history of operations, enabling agents to access and reuse data efficiently.
* **Client-Server Communication**: The Streamlit application (client) communicates with a Flask-based API backend (server) to handle the computationally intensive tasks.

## Prerequisites and Setup

1.  **Install Python libraries**:
    Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Configuration**:
    * Create a `key.env` file in the project's root directory.
    * Add the necessary environment variables, including your OpenAI API key and PostgreSQL database connection details.
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    PG_USER="your_pg_user"
    PG_PASSWORD="your_pg_password"
    PG_HOST="your_pg_host"
    PG_PORT="5432"
    PG_DATABASE="your_pg_database"
    ```
    * Ensure you have a running PostgreSQL database accessible with the provided connection details.

## How to Run the Project

1.  **Start the Backend Server**:
    Open a terminal and run the `main.py` file to start the Flask API server.
    ```bash
    python main.py
    ```
    The server will run on `http://0.0.0.0:8000`.

2.  **Launch the Frontend**:
    Open another terminal and run the `app.py` file to launch the Streamlit application.
    ```bash
    streamlit run app.py
    ```
