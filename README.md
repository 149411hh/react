# Research Agent

一个基于 ReAct 框架的深度研究型自主 AI Agent，能自主上网搜索、浏览网页、交叉验证信息，解决复杂多跳事实性问题。

## 核心功能
- 多轮自主推理与工具调用（支持 8-15 轮深度搜索）
- 双引擎搜索（Google + Bing），支持中英双语
- 智能网页内容提取与总结
- 严格的答案验证和格式控制
- 批量评测系统（eval.py）

## 技术栈
- **模型**：Qwen3.5-plus (阿里云 DashScope)
- **框架**：自定义 ReAct Agent 循环
- **工具**：Serper Google 搜索、IQS Bing 搜索、Jina Reader
- **其他**：FastAPI、异步处理

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt

配置环境变量（复制 .env.example 为 .env 并填入密钥）
DASHSCOPE_API_KEY
SERPER_API_KEYS
JINA_API_KEYS

启动服务Bashuvicorn agent:app --reload
批量评测Bashpython eval.py 0 50 my_results.jsonl

项目结构

agent_loop.py — 核心 ReAct 循环
prompts.py — 系统提示词与格式控制
tools_search.py / tools_visit.py — 搜索与网页工具
eval.py — 自动化评测脚本
skills/ — 可扩展 Agent Skills