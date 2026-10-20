import json
from datetime import datetime
from openai import OpenAI
import os
import chromadb
from  chromadb.utils import embedding_functions

from dotenv import load_dotenv
# === 全局初始化：uvicorn启动时之跑一次 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录 practice/
CHROMA_DIR = os.path.join(BASE_DIR, "..", "chroma_db")  # 指向项目根的 chroma_db
db = chromadb.PersistentClient(path=CHROMA_DIR)
bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
   model_name="BAAI/bge-small-zh-v1.5" )
table = db.get_or_create_collection("text_index_bge2",embedding_function= bge_ef)
print(">>> 知识库已连接，共", table.count(), "条")


load_dotenv()
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


if __name__ == '__main__':
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com")
    ques = input("请输入问题")
    messages=[
            {"role":"system","content":"你是一个政务问答助手"},
            {"role":"user","content":ques}
        ]
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages=messages,
        tools=tools
    )
    msg = response.choices[0].message
    print(msg.model_dump())
    if msg.tool_calls:
        messages.append(msg)
        for i in msg.tool_calls:
            print("调用工具：",i.function.name)
            if i.function.name == "get_current_time":
                currentTime=get_current_time()
                messages.append({"role": "tool", "tool_call_id": i.id, "content": str(currentTime)})
            if i.function.name == "search_knowledge_base":
                args = json.loads(i.function.arguments)
                searchResult = search_knowledge_base(args["query"])
                messages.append({"role": "tool", "tool_call_id": i.id, "content": str(searchResult)})

        responseResult = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        print(responseResult.choices[0].message.content)
    else:
        print(msg.content)

