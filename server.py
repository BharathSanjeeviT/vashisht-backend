import pandas as pd
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from os import environ
from sqlalchemy import create_engine, text

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.core.query_engine.pandas import PandasInstructionParser
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = "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"

engine = create_engine(f'postgresql://postgres.msuzsakkwbftgtxwuoim:fbXhdVUNtIE2fCTt@aws-0-ap-south-1.pooler.supabase.com:5432/postgres')

df = pd.read_csv("https://raw.githubusercontent.com/pranavs6/V_hack/main/ex03.csv")
print(df)
instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)
pandas_prompt_str = ( "You are working with a pandas dataframe in Python.\n" "The name of the dataframe is `df`.\n" "This is the result of `print(df.head())`:\n" "{df_str}\n\n" "Follow these instructions:\n" "{instruction_str}\n" "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, return only the id value in array format from the query results if not return a empty array.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = OpenAI(model="gpt-3.5-turbo")

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")
keys = ['id', 'name', 'img_url', 'price', 'description']

@app.route("/query", methods=["POST"])
def query():
    connection = engine.connect()
    try:
        req_data = request.get_json()
        query = req_data['query']
        model_res = qp.run( query_str = query )
        ids = model_res.message.content
        if(len(ids) > 2):
            d = str(ids)
            d =  '('+d[1:-1]+')'
            sql_query = text("SELECT * FROM products WHERE id IN "+d)
            print(sql_query)
            result = connection.execute(sql_query)
        else:
            print("Fuckkkk")
            result =[]
        data = []
        for i in result:
            new_dict = {}
            for j in range(len(i)):
                new_dict[keys[j]]= i[j]
            data.append(new_dict)
        connection.close()
        return jsonify(data)
    except Exception as e:
        connection.close()
        return jsonify({'error': str(e)})


@app.route("/add_data", methods=["POST"])
def add_data():
    try:
        req_data = request.get_json()
        id = len(df)+1
        category = req_data.get('category', '')
        color = req_data.get('color', '')
        size = req_data.get('size', '')
        price = req_data.get('price', '')
        description = req_data.get('description', '')

        # Append new data to DataFrame
        df.loc[len(df)] = [id, category, color, size, price, description]

        return jsonify({'message': 'Data added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
