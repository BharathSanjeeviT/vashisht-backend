import pandas as pd
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from os import environ
from sqlalchemy import create_engine, text
from io import StringIO
from google.cloud import storage
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

from supabase import create_client, Client
url="supabase:url:here"
key="supabas:key:here"
supabase: Client = create_client(url, key)

res_data = supabase.table('products').select("*").csv().execute()
dataset=res_data.data
print(dataset)
csv_data=StringIO(dataset)
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

engine = create_engine(f'pg:url')

df = pd.read_csv(csv_data)
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

print(df.columns)

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
            sql_query = text("SELECT id, name, img_url, price, description FROM products WHERE id IN "+d)
            print(sql_query)
            result = connection.execute(sql_query)
        else:
            print("Not_Found")
            result =[]
        data = []
        print(result)
        print()
        print(keys)
        for i in result:
            print(i)
            new_dict = {}
            for j in range(len(i)):
                new_dict[keys[j]]= i[j]
            data.append(new_dict)
        print(data)
        connection.close()
        return jsonify(data)
    except Exception as e:
        connection.close()
        return jsonify({'error': str(e)})


@app.route("/add_data", methods=["POST"])
def add_data():
    try:
        req_data = request.get_json()
        idx = len(df)+1
        id = len(df)+1
        category = req_data.get('category', '')
        color = req_data.get('color', '')
        size = req_data.get('size', '')
        price = req_data.get('price', '')
        description = req_data.get('description', '')
        name = req_data.get('name','')
        retailer = req_data.get('retailer','')
        img_data = req_data.get('img_data','')
        image=base64.b64decode(img_data)
        

        try:
            response = supabase.storage.from_("product-images").update(file=image, path="f"+str(idx)+".jpg", file_options={"content-type": "image/jpeg"})
        except:
            response = supabase.storage.from_("product-images").upload(file=image, path="f"+str(idx)+".jpg", file_options={"content-type": "image/jpeg"})
        res = supabase.storage.from_('product-images').get_public_url('f'+str(idx)+'.jpg')
        print(res)
        link=str(res)
        df.loc[len(df)] = [id, name, link, price, description, retailer, color, size, category]

        response = supabase.table('products').insert({'id':idx,'name':name,'price':price,'description':description,'retailer':retailer,'color':color,'img_url':link,'category':category,'size':size}).execute()

        return jsonify({'message': 'Data added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/retailers/signup", methods=["POST"])
def signup():
    try:
        req_data = request.get_json()
        email = req_data.get('email', '').lower()
        name = req_data.get('name', '')
        password = req_data.get('password', '')
        location = req_data.get('location', '')
        supabase.table('Retailers').insert({"email": email, "name": name,"location":location,"password":password}).execute()
        return jsonify({'message': 'Signup successful'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/retailers/login", methods=["POST"])
def login():
    try:
        req_data = request.get_json()
        email = req_data.get('email', '').lower()
        password = req_data.get('password', '')
        response = supabase.table('Retailers').select("*").eq('email', email).eq("password", password).execute()
        data = response.data
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
