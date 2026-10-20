from datetime import datetime

import openai
from fastapi import FastAPI
import os
from dotenv import load_dotenv
import chromadb
from  chromadb.utils import embedding_functions
from openai import OpenAI
import json
from fastapi.responses import FileResponse


load_dotenv()
app = FastAPI(title="政务智能助手 API", version="0.1")


# === 全局初始化：uvicorn启动时之跑一次 ===
db = chromadb.PersistentClient(path="./chroma_db")
bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
   model_name="BAAI/bge-small-zh-v1.5" )
table = db.get_or_create_collection("text_index_bge2",embedding_function= bge_ef)
client = OpenAI(api_key= os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

tools = [
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"获取当前日期和时间，当用户问今天几号、现在的几点、星期几时调用",
            "parameters":{"type":"object","properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"检索政务知识库，当用户咨询办事流程、申报步骤、政策、平台操作等政务资料问题 时调用",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{"type":"string","description":"要在知识库中检索的问题或 关键词"}
                },
                "required":["query"]
            }
        }
    }
]

def get_current_time():
    current_time = datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M:%S 星期%w")

def search_knowledge_base(query):
    results = table.query(query_texts=[query],n_results=2)
    return "\n\n".join(results["documents"][0])


@app.get("/ask")
def ask(q:str):

    result = table.query(query_texts=[q],n_results=2)
    context = f"""用户的问题：{q}
    参考内容：（1）{result["documents"][0][0]}；
    （2）{result["documents"][0][1]}；
    如果参考资料里面没有答案，请回复无法回答"""
    chat = client.chat.completions.create(model="deepseek-chat",
                                          messages=[
                                              {"role":"system","content":"你是一个政务问答系统"},
                                              {"role":"user", "content": context}
                                                                          ]
                                          )
    return {"question": q,
            "answer": chat.choices[0].message.content,
            "sources": result["metadatas"][0]}


@app.get("/agent")
def get_agent(query: str):
    messages = [
        {"role": "system", "content": "你是一个政务问答助手"},
        {"role": "user", "content": query}
    ]
    trace = []
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools)
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            if call.function.name == "get_current_time":
                trace.append("get_current_time")
                result = get_current_time()
            elif call.function.name == "search_knowledge_base":
                trace.append("search_knowledge_base")
                args = json.loads(call.function.arguments)
                result = search_knowledge_base(args["query"])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        final = client.chat.completions.create(model="deepseek-chat", messages=messages)
        answer = final.choices[0].message.content
    else:
        answer = msg.content

    return {"question": query, "answer": answer, "tools_called": trace}

@app.get("/")
def home():
    return {"message": "服务运行正常"}


@app.get("/hello")
def hello(name: str = "世界"):
    # 访问时试试 /hello?name=刘某
    return {"你好": name}

@app.get("/ping")
def get_ping(pong: str = "pong"):
    return {"ping": pong}

@app.get("/ui")
def ui():
    return FileResponse("index.html")
