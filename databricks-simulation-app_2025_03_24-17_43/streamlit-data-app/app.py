import os
from databricks import sql
from databricks.sdk.core import Config
import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

def score_model():
    url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/agents_dkushari_uc-dbappsdemo-basic_rag_demo/invocations'
    headers = {'Authorization' : 'Bearer {}'.format(dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()), 'Content-Type': 'application/json'}
    #ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = {
                  "messages": [
              {
                  "role": "user",
                  "content": "What is Retrieval-augmented Generation?"
              }
            ]
    }
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def sqlQuery(query: str) -> pd.DataFrame:
    cfg = Config() # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

st.set_page_config(layout="wide")

@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def getData():
    # This example query depends on the nyctaxi data set in Unity Catalog, see https://docs.databricks.com/en/discover/databricks-datasets.html for details
    return sqlQuery("select * from dkushari_uc.dbappsdemo.sample_questions;")

data = getData()

st.header("Databricks simulation App !!!")
endpoints = w.serving_endpoints.list()
filter_criteria = "agents_dkushari_uc"
endpoint_names = [endpoint.name for endpoint in endpoints if endpoint.name.startswith(filter_criteria)]

st.subheader("Select a model served by Model Serving Endpoint from the drop down")
selected_model = st.selectbox(
    label="Select your model:", options=endpoint_names
)

temperature = st.slider(
        "Select temperature:",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the randomness of the LLM output. Only applicable for chat/completions queries.",
    )

st.subheader("Select your question for simulation...")
options = data.apply(lambda row: f"{row['id']}: {row['question']}", axis=1).tolist()
selected_question = st.radio("Select a question:", options)
st.write(f"You selected: {selected_question}")

if st.button("Get Answer"):
    try:
        response = w.serving_endpoints.query(name=selected_model, messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="You are a helpful assistant. Only answer questions related to Databricks and for all other questions respond saying you are not aware.",
                ),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=selected_question,
                ),
            ],
            temperature=temperature)
        # response = score_model()
        
        choices = response.as_dict().get("choices", [])
        # choices = response.get("messages", [])
        if choices:
            st.subheader("Model Response:")
            message_content = choices[0].get("message", {}).get("content", "")
            st.markdown(message_content)
    except Exception as e:
        st.error(f"Error querying model: {e}")


