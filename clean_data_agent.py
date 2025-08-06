import os
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from langchain_core.tools import Tool
# Removed PythonAstREPLTool import as it's not used here
from langchain_openai import ChatOpenAI
# Removed SQLDatabase import as it's not directly needed for the general agent
from langchain.agents import AgentExecutor, create_openai_tools_agent # Import necessary components
from dotenv import load_dotenv
# Use a standard Hub prompt or create a compatible one
# from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import prompt components
from langchain.memory import ConversationBufferMemory # Import memory


load_dotenv()

# --- DATABASE CONNECTION SETUP ---
db_user = os.getenv("PG_USER")
db_password = os.getenv("PG_PASSWORD")
db_host = os.getenv("PG_HOST")
db_port = os.getenv("PG_PORT", "5432")
db_name = os.getenv("PG_DATABASE")
if not all([db_user, db_password, db_host, db_name]):
    raise ValueError("Missing one or more PostgreSQL connection environment variables")
db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_uri)
# Test connection
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {e}")
    exit() # Exit if connection fails

# --- PROFILING FUNCTIONS ---
# Keep your profiling functions as they are, they correctly use the engine

def list_tables(*args, **kwargs):
    """Return a list of table names in the connected database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        return f"Error listing tables: {e}"


def profile_table(table_name: str) -> dict:
    """
    Profile a single table: total_rows, per-column (null_rate, distinct_count),
    numeric (min, max, mean, std), string (avg_length).
    """
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
             return f"Error: Table '{table_name}' not found."
        cols = inspector.get_columns(table_name)
        profile = {"table": table_name, "columns": {}}
        with engine.connect() as conn:
            total_rows_result = conn.execute(text(f"SELECT COUNT(*) FROM \"{table_name}\"")) # Quote table name
            total = total_rows_result.scalar()

            for col in cols:
                name = col["name"]
                dtype = col["type"]
                stats = {}
                # Quote column name for safety
                quoted_name = f'"{name}"'

                # Null rate
                null_q = f"SELECT COUNT(*) FROM \"{table_name}\" WHERE {quoted_name} IS NULL"
                null_count = conn.execute(text(null_q)).scalar()
                stats['null_rate'] = null_count / total if total > 0 else 0

                # Distinct count
                distinct_q = f"SELECT COUNT(DISTINCT {quoted_name}) FROM \"{table_name}\""
                stats['distinct_count'] = conn.execute(text(distinct_q)).scalar()

                # Numeric stats (check type more robustly)
                # Using str(dtype) comparison is common for SQLAlchemy types
                dtype_str = str(dtype).upper()
                if 'INT' in dtype_str or 'FLOAT' in dtype_str or 'DECIMAL' in dtype_str or 'NUMERIC' in dtype_str:
                    # Ensure column is actually numeric before aggregations
                    try:
                        num_q = f"SELECT MIN({quoted_name}), MAX({quoted_name}), AVG({quoted_name}::numeric), STDDEV_POP({quoted_name}::numeric) FROM \"{table_name}\""
                        mn, mx, avg, sd = conn.execute(text(num_q)).fetchone()
                        stats.update({ 'min': mn, 'max': mx, 'mean': float(avg) if avg else None, 'std': float(sd) if sd else None }) # Cast Decimal to float for JSON serialization if needed
                    except Exception as num_e:
                        print(f"Warning: Could not calculate numeric stats for {table_name}.{name}: {num_e}")
                        stats.update({ 'min': None, 'max': None, 'mean': None, 'std': None })


                # String stats (check type more robustly)
                elif 'CHAR' in dtype_str or 'TEXT' in dtype_str or 'VARCHAR' in dtype_str:
                     try:
                        len_q = f"SELECT AVG(LENGTH({quoted_name})) FROM \"{table_name}\" WHERE {quoted_name} IS NOT NULL"
                        avg_len_result = conn.execute(text(len_q)).scalar()
                        stats['avg_length'] = float(avg_len_result) if avg_len_result else None
                     except Exception as str_e:
                        print(f"Warning: Could not calculate string stats for {table_name}.{name}: {str_e}")
                        stats['avg_length'] = None

                profile['columns'][name] = stats
        profile['total_rows'] = total
        return profile
    except Exception as e:
        return f"Error profiling table {table_name}: {e}"
    
def get_sample_data(table_name: str, num_rows: int = 10) -> str:
    """
    Fetches a small sample of rows (default 10) from the specified table.
    Returns the data as a string (e.g., CSV or DataFrame string representation).
    """
    try:
        # Ensure num_rows is reasonable
        num_rows = max(1, min(num_rows, 100)) # Limit sample size
        query = text(f'SELECT * FROM "{table_name}" LIMIT :limit') # Use parameter binding
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={'limit': num_rows})
        # Return as string - adjust format as needed (e.g., df.to_csv(index=False))
        return df.to_string()
    except Exception as e:
        return f"Error fetching sample data from {table_name}: {e}"
    
def execute_sql(sql_query: str) -> str:
    """
    Executes a given SQL query against the database.
    USE WITH EXTREME CAUTION, ESPECIALLY FOR UPDATE/DELETE/ALTER.
    Consider adding safety checks or requiring user confirmation.
    Returns a success message, number of rows affected, or an error message.
    """
    # *** SAFETY MECHANISM EXAMPLES - CHOOSE/COMBINE WISELY ***

    # 1. Log Only (Safest): Comment out the execution part initially
    print(f"--- SQL TO EXECUTE (LOGGED ONLY) ---\n{sql_query}\n------------------------------------")
    # return "SQL query logged. Execution disabled for safety." # Use this during development/testing

    # 2. Prevent Modification Statements (Safer):
    # if any(keyword in sql_query.upper() for keyword in ["UPDATE ", "DELETE ", "INSERT ", "ALTER ", "DROP ", "CREATE ", "TRUNCATE "]):
    #     return "Error: SQL query appears to be a modification statement and execution is restricted for safety."

    # 3. Execute with Transaction (Allows Rollback on Error):
    try:
        with engine.connect() as conn:
            with conn.begin() as transaction: # Start transaction
                try:
                    # Check if it's a SELECT query to potentially fetch results
                    is_select = sql_query.strip().upper().startswith("SELECT")
                    result = conn.execute(text(sql_query))
                    if is_select:
                        # For SELECT, you might want to return fetched data (limited)
                        # fetched_data = result.fetchmany(20) # Fetch up to 20 rows
                        # transaction.commit() # Commit SELECT transaction
                        # return f"Query executed successfully. Fetched data:\n{fetched_data}"
                        return "SELECT query executed successfully (results not returned by this tool)."
                    else:
                        # For DML (UPDATE, INSERT, DELETE), return row count
                        row_count = result.rowcount
                        transaction.commit() # Commit changes if no error
                        return f"Query executed successfully. Rows affected: {row_count}"
                except Exception as e:
                    print(f"Error during query execution, rolling back transaction: {e}")
                    transaction.rollback() # Rollback on error
                    return f"Error executing query: {e}. Transaction rolled back."
    except Exception as e:
        # Errors connecting or starting transaction
        return f"Database connection or transaction error: {e}"
    
# Wrap profiling functions as LangChain Tools
list_tables_tool = Tool(
    name="list_tables",
    func=list_tables,
    description="List all table names in the connected PostgreSQL database."
)
profile_table_tool = Tool(
    name="profile_table",
    func=profile_table,
    description="Generate profiling statistics for a specific table. Input should be the table name (string)." # Added input guidance
)

get_sample_data_tool = Tool(
    name="get_sample_data",
    func=get_sample_data,
    description="Fetches a small sample of rows (default 10, max 100) from a specified table name to inspect actual data values. Input is the table name (string) and optionally the number of rows (integer)."
)



execute_sql_tool = Tool(
    name="execute_sql",
    func=execute_sql,
    description=(
        "Executes a given SQL query against the database. "
        "CRITICAL: Be absolutely sure the query is correct and safe, especially for UPDATE, DELETE, or ALTER statements as it can modify data. "
        "Avoid generating queries that could drop tables or cause significant data loss unless specifically instructed and confirmed. "
        "Input is the SQL query string."
    )
)
# --- AGENT SETUP ---

llm = ChatOpenAI( temperature=0, model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))

# Define the memory
# memory_key="chat_history" means the history will be injected into the prompt variable "chat_history"
# return_messages=True ensures the memory stores Message objects, compatible with ChatPromptTemplate
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# List of tools the agent can use
tools = [list_tables_tool, profile_table_tool, get_sample_data_tool, execute_sql_tool]

# Create a prompt template suitable for OpenAI Tools agen
# This typically includes placeholders for input, agent_scratchpad, and optionally chat_history
# The agent_scratchpad is where previous agent steps (tool calls and observations) are stored
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant designed to help with cleaning and preprocessing PostgreSQL database tables. Your goal is to:\n"
            "1. Understand the user's request for cleaning a specific table or addressing specific data quality issues.\n"
            "2. Use the available tools (`list_tables`, `profile_table`, `get_sample_data`) to thoroughly analyze the target table(s).\n"
            "3. Based on the analysis and user request, formulate a clear, step-by-step data cleaning strategy. Explain *why* each step is needed (e.g., 'High null rate in column X suggests imputation').\n"
            "4. **IMPORTANT**: Present this strategy to the user for review and explicit approval BEFORE proceeding with any modifications.\n"
            "5. If (and only if) the user approves the strategy, translate each step into precise and safe SQL queries.\n"
            "6. Use the `execute_sql` tool to execute these SQL queries one by one. Be extremely careful with UPDATE, DELETE, and ALTER statements.\n"
            "7. Report the outcome of each execution step.\n"
            "Always prioritize data safety. If unsure about a step or a query, ask for clarification or user confirmation."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)



# Create the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the Agent Executor, which runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, # Pass the memory object here!
    handle_parsing_errors=True # Still useful for general robustness
)


    # --- INTERACTIVE CHAT LOOP ---
if __name__ == "__main__":
    print("Data Cleaning Agent Initialized. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break

            # Invoke the agent executor
            # The executor will automatically:
            # 1. Load history from `memory`
            # 2. Format the prompt with history and current input
            # 3. Run the agent logic (LLM calls, tool calls)
            # 4. Get the final response
            # 5. Save the current input and final response to `memory`
            response = agent_executor.invoke({"input": user_input})

            print(f"Agent: {response['output']}")

        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally break the loop on error or add more robust handling
            # break