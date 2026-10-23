import time
from datetime import datetime

from fastapi import FastAPI
import os
from dotenv import load_dotenv
import chromadb
from  chromadb.utils import embedding_functions
from openai import OpenAI, APITimeoutError,APIConnectionError,APIError,RateLimitError
import json
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse

load_dotenv()
app = FastAPI(title="政务智能助手 API", version="0.1")


# === 全局初始化：uvicorn启动时之跑一次 ===
db = chromadb.PersistentClient(path="./chroma_db")
bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
   model_name="BAAI/bge-small-zh-v1.5" )
table = db.get_or_create_collection("text_index_bge2",embedding_function= bge_ef)
client = OpenAI(api_key= os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                timeout=30)

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

class LLMError(Exception):
    pass

def get_current_time():
    current_time = datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M:%S 星期%w")

def search_knowledge_base(query):
    results = table.query(query_texts=[query],n_results=2)
    return "\n\n".join(results["documents"][0])

def get_chat(messages,tools:object=[],tool:bool=False,model:str = "deepseek-chat",max_retries:int=3):
    for attempt in range(max_retries):
        try:
            #以下参考https://blog.csdn.net/Qnesp/article/details/159853262中的错误 处理最佳实践
            if tool:
                result =  client.chat.completions.create(
                    model=model, messages=messages, tools=tools)
            else:
                result = client.chat.completions.create(
                    model=model, messages=messages
                )
            return result
        except RateLimitError as e:
            # 频率限制错误：指数退避
            wait_time =2**attempt
            print(f"达到速率限制，{wait_time}秒后重试")
            time.sleep(wait_time)
        except APITimeoutError as e:
            # 超时：快速重试
            print(f"请求超时。重试中...")
            time.sleep(1)
        except APIConnectionError as e:
            # 网络问题：稍长等待
            print(f"网络连接错误，3秒后重试...")
            time.sleep(3)
        except APIError as e:
            # 其他API错误
            print(f"API错误：{e}")
    print("达到最大重试次数，请求失败")
    raise LLMError(f"LLM 调用失败，已重试{max_retries}次")

@app.exception_handler(LLMError)
async def llm_error_handler(request,exc):
    return JSONResponse(statue_code=503,content={"error":"llm_unavailadle","detail":str(exc)})





@app.get("/ask")
def ask(q:str,n:int=2):
    result = table.query(query_texts=[q],n_results=n)
    context = f"""用户的问题：{q}
        如果参考资料里面没有答案，请回复无法回答
        参考内容："""
    if result["documents"][0]:
        for i,r in enumerate(result["documents"][0]):
            context += f"({i+1})"+r+"\n"
    else:
        context += "未检索到相关内容"

    chat = get_chat(messages=[{"role":"system","content":"你是一个政务问答系统"},{"role":"user", "content": context}])
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
    k=1
    while k:
        resp = get_chat(model="deepseek-chat", messages=messages,tool=True, tools=tools)
        msg = resp.choices[0].message
        k=0
        if msg.tool_calls:
            messages.append(msg)
            result =[]
            for call in msg.tool_calls:
                if call.function.name == "get_current_time":
                    trace.append("get_current_time")
                    result.append(get_current_time())
                elif call.function.name == "search_knowledge_base":
                    trace.append("search_knowledge_base")
                    try:
                        args = json.loads(call.function.arguments)
                        result.append(search_knowledge_base(args["query"]))
                    except Exception as e:
                        k=1
                        trace = []
                        messages.append({"role": "user", "content": e})
                        continue
            for i,call in enumerate(msg.tool_calls):
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result[i])})
            final =get_chat(model="deepseek-chat", messages=messages)
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
