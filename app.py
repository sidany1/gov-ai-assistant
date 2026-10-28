import time
from datetime import datetime
from pathlib import Path

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
# 用 __file__ 定位，数据库路径和"在哪个目录启动"无关
DB_PATH = Path(__file__).resolve().parent / "chroma_db"
db = chromadb.PersistentClient(path=str(DB_PATH))
bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
   model_name="BAAI/bge-small-zh-v1.5" )
table = db.get_or_create_collection("text_index_bge2",embedding_function= bge_ef)
client = OpenAI(api_key= os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                timeout=30,
                max_retries=1)

MAX_STEPS=5
MAX_DISTANCE = 0.6

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
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = "一二三四五六日"
    current_week = s[datetime.now().weekday()]
    return str(current_time)+" 星期"+str(current_week)

def search_knowledge_base(query,n:int=2):
    query_results = table.query(query_texts=[query],n_results=n)
    results =[]
    print(query_results)
    for meta,dist,docu in zip(query_results["metadatas"][0],query_results["distances"][0],query_results["documents"][0]):
        if dist<MAX_DISTANCE:
            result ={"text": docu, "source": meta["source"], "page": meta["page"], "distance": dist}
            results.append(result)
    if not results:
        return [{"text": "未检索到相关信息", "source": None, "page": None, "distance": None}]

    return results

def get_chat(messages,tools=None,model:str = "deepseek-chat",max_retries:int=3):
    last_reason = "unknown"
    for attempt in range(max_retries):
        try:
            if tools:
                return client.chat.completions.create(
                    model=model, messages=messages, tools=tools)
            else:
                return client.chat.completions.create(
                    model=model, messages=messages
                )
        except RateLimitError as e:
            # 频率限制错误：指数退避
            last_reason = "RateLimitError"
            wait_time =2**attempt
            print(f"达到速率限制，{wait_time}秒后重试")
            time.sleep(wait_time)
        except APITimeoutError as e:
            # 超时：快速重试
            last_reason = "APITimeoutError"
            print(f"请求超时。重试中...")
            time.sleep(1)
        except APIConnectionError as e:
            # 网络问题：稍长等待
            last_reason = "APIConnectionError"
            print(f"网络连接错误，3秒后重试...")
            time.sleep(3)
        except APIError as e:
            # 其他API错误
            print(f"API错误：{e}")
            raise LLMError(f"模型调用失败：{e}" ) from e
    print("达到最大重试次数，请求失败")
    raise LLMError(f"LLM 调用失败，已重试{max_retries}次，最后问题{last_reason}")


def run_tool(call):
    try:
        if call.function.name =="get_current_time":
            return get_current_time()
        if call.function.name == "search_knowledge_base":
            argu = json.loads(call.function.arguments)
            query = argu["query"]
            results = search_knowledge_base(query)
            if not results[0]["distance"]:
                return results[0]["text"]
            return "\n\n".join(f'（{n}）{res["text"]}\n来源：{res["source"]},页数：{res["page"]},距离:{res["distance"]}' for n,res in enumerate(results))

        return f"{call.function.name}工具名不存在"
    except json.JSONDecodeError:
        return (
            f"该参数不是合法的json，收到的是：{call.function.arguments!r},请提供合法的json"
        )
    except Exception as e:
        return f"错误：工具执行失败：{e}"

@app.exception_handler(LLMError)
async def llm_error_handler(request,exc:LLMError):
    return JSONResponse(status_code=503,content={"error":"服务暂时不可用 ，请过段时间重试","detail":str(exc)})





@app.get("/ask")
def ask(q:str,n:int=2):

    result = search_knowledge_base(q,n)
    if not result[0]["distance"]:
        return {"question": q,
            "answer": result[0]["text"],
            "sources": None}
    referenceContext ="\n\n".join(f'（{n}）{res["text"]}\n来源：{res["source"]},页数：{res["page"]},距离:{res["distance"]}' for n,res in enumerate(result))

    context = f"""用户的问题：{q}
        如果参考资料里面没有答案，请回复无法回答
        参考内容：{referenceContext}"""
    messages = [{"role": "system", "content": "你是一个政务问答系统"},
                {"role": "user", "content": context}]
    chat = get_chat(messages=messages)
    return {"question": q,
            "answer": chat.choices[0].message.content,
            "sources": referenceContext}


@app.get("/agent")
def get_agent(query: str):
    messages = [
        {"role": "system", "content": "你是一个政策规划问答助手"},
        {"role": "user", "content": query}
    ]
    trace = []
    for max in range(MAX_STEPS):
        resp = get_chat(model="deepseek-chat", messages=messages, tools=tools)
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                result = run_tool(call)
                trace.append(call.function.name)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        else:
            return {"question": query, "answer": msg.content, "tools_called": trace,"llm_rounds": max + 1}

    raise LLMError(f"超过最大步数 {MAX_STEPS}，模型可能陷入循环")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8000, reload=True)