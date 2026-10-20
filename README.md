定位：基于FastAPI的政务智能助手，演绎rag与Agent的区别
技术栈：FastAPI、openai、
主要为/ask和/agent两个接口，/ask接口可以将问题向量化放入向量数据库进行匹配，然后交给大模型回答。而/agent提供大模型多个工具，允许大模型自行判断是否选择工具、选什么工具辅助进行回答。
启动方式：在命令行运行uvicorn app:app --reload
![img.png](img.png)