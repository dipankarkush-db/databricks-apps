import os
from databricks import sql
from databricks.sdk.core import Config
import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from json import loads

w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# def score_model():
#     url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/agents_dkushari_uc-dbappsdemo-basic_rag_demo/invocations'
#     headers = {'Authorization' : 'Bearer {}'.format(dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()), 'Content-Type': 'application/json'}
#     #ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
#     data_json = {
#                   "messages": [
#               {
#                   "role": "user",
#                   "content": "What is Retrieval-augmented Generation?"
#               }
#             ]
#     }
#     response = requests.request(method='POST', headers=headers, url=url, json=data_json)
#     if response.status_code != 200:
#         raise Exception(f'Request failed with status {response.status_code}, {response.text}')
#     return response.json()

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

traditional_model_input ='[{"sepal length (cm)": 6.2, "sepal width (cm)": 3.4, "petal length (cm)": 5.4, "petal width (cm)": 2.3}]'

st.header("Databricks simulation App !!!",divider=True)
st.markdown("""### Choose Model type""")
model_type = st.radio(label="Choose between Traditional ML and LLM...", options=["Traditional ML", "Large Language Model"])
if model_type == "Traditional ML":
    endpoints = w.serving_endpoints.list()
    filter_criteria = "dkushari-uc-demoapps-traditional"
    endpoint_names = [endpoint.name for endpoint in endpoints if endpoint.name.startswith(filter_criteria)]
    
    st.subheader("Select a model served by Model Serving Endpoint from the drop down")
    selected_model = st.selectbox(
        label="Select your model:", options=endpoint_names
    )
    response_dict = {}
    species_info = {
    0: {"species": "Iris-setosa", "description": "Small petals, easily distinguishable"},
    1: {"species": "Iris-versicolor", "description": "Medium-sized petals, intermediate characteristics"},
    2: {"species": "Iris-virginica", "description": "Largest petals, difficult to distinguish from Versicolor"}
    }

    st.markdown("""
    ##### Example Inputs:
    - [{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}]
    - [{"sepal length (cm)": 6.0, "sepal width (cm)": 2.9, "petal length (cm)": 4.5, "petal width (cm)": 1.5}]
    - [{"sepal length (cm)": 6.9, "sepal width (cm)": 3.1, "petal length (cm)": 5.4, "petal width (cm)": 2.3}]
    """)
    
    input_value = st.text_area(
            "Enter model input",
            placeholder=traditional_model_input,
        )
    if st.button("Get Prediction"):
        response = w.serving_endpoints.query(
            name=selected_model, dataframe_records=loads(input_value)
        )
        response_dict=response.as_dict()
        predictions = response_dict["predictions"]
        data = []
        for prediction in predictions:
            species_name = species_info.get(prediction, {}).get("species", "Unknown")
            description = species_info.get(prediction, {}).get("description", "No description available")
            data.append([prediction, species_name, description])

        # Create a DataFrame for the table
        df = pd.DataFrame(data, columns=["Class Label", "Species Name", "Description"])

        # Streamlit table display
        st.write("##### Iris Model Prediction Result")
        st.table(df)
elif model_type == "Large Language Model":
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


