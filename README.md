Research Agent 项目解析
这是一个高度优化、专为复杂信息检索和研究任务设计的自主 Web Research Agent（网络信息寻求大师）。它基于 ReAct（Reason + Act） 框架构建，使用 Qwen3.5-plus 大模型，通过工具调用实现深度、多轮、持久的互联网信息搜索与验证。
项目核心目标：在复杂、多跳（multi-hop）问题上实现高准确率，强调问题分解、持久搜索、交叉验证和答案规范化，适合用于评测或实际研究场景。

项目整体架构
texteval.py          ← 批量评测脚本（question.jsonl → my_results.jsonl）
agent.py         ← FastAPI 服务端（提供 / 接口，支持 SSE 流式）
agent_loop.py    ← 核心 ReAct Agent 循环（最重要文件）
prompts.py       ← 系统提示词 + 用户提示词模板 + 提取器提示词
tools_search.py  ← 搜索工具（Serper Google + IQS Bing）
tools_visit.py   ← 网页访问与内容提取工具（Jina Reader + 后备 httpx + LLM 总结）
skills.py        ← Agent Skills 扩展系统（可选插件化能力）

核心设计理念与亮点
1. 强制性问题分解 + 深度搜索策略

系统提示要求：每题必须先分解成子问题，列出所有实体、约束、关系。
鼓励 8-15 轮交互，明确禁止“少于 5 轮就放弃”。
支持中英双语搜索（Google 适合国际/英文，Bing/IQS 适合中文内容）。

2. 严格的答案验证机制

强制交叉验证：最终答案前至少 2 个独立来源确认。
约束校验：回答前必须重新阅读原题，检查是否满足每一个约束条件。
信任多源共识：即使细节有微小差异（如年份差 1-2 年、拼写变体），只要多源指向同一答案，就倾向采信。

3. 答案格式极致控制

必须精确匹配参考答案（字符级对比）。
语言规则非常严格：
中文问题 → 中文答案（使用标准中文译名）
英文问题 → 英文答案
要求全名（姓+名）、官方名称、特定格式时严格遵守

使用 <think> + <answer> 标签强制结构化输出。

4. 鲁棒性与容错设计

超时保护（9 分钟）、Token 限制（约 50万 tokens）
内容过滤器处理（data_inspection_failed 时自动重定向或净化）
早答阻断（前 3 轮过快回答会被要求额外验证）
工具结果截断、上下文管理
多轮失败后强制汇总已有信息输出答案


主要组件详解
prompts.py

SYSTEM_PROMPT：定义 Agent 身份（Web Information Seeking Master） + 6 大核心原则。
USER_PROMPT_TEMPLATE：包含工具定义（search、visit）、输出格式要求、语言与命名规范。
EXTRACTOR_PROMPT：用于网页内容提取的 LLM 提示，要求输出 JSON（rational、evidence、summary）。

agent_loop.py（核心引擎）

实现完整的 ReAct 循环：think → tool_call → tool_response → ... → answer
支持 <tool_call> JSON 格式和旧式 <function=...> 格式兼容
使用 tiktoken 精确计算 token
_force_answer()：达到上限时强制总结输出
丰富的调试输出（轮次、耗时、token 数、输出预览）

工具层
tools_search.py：

双引擎支持：Serper（Google） + IQS（阿里云类 Bing 搜索）
自动根据查询语言选择引擎（含中文检测）
差结果自动跨引擎回退 + 查询简化
结果格式统一，美观且信息丰富（标题、链接、片段、日期、来源）

tools_visit.py：

优先使用 Jina Reader（干净 Markdown）
失败回退到 httpx + BeautifulSoup
使用 LLM（qwen-plus）按用户 goal 智能提取证据和总结
并发访问多个网页

评测系统 (eval.py)

支持从 question.jsonl 批量运行指定 ID 范围
自动跳过已完成题目（断点续跑）
答案归一化对比（大小写、数字类型、strip）
输出详细准确率统计（单次运行 + 整体累计）


技术栈

模型：Qwen3.5-plus（阿里云 DashScope）
框架：FastAPI + AsyncOpenAI
搜索：Serper.dev（Google） + 阿里云 IQS
网页阅读：Jina.ai Reader + BeautifulSoup
其他：python-dotenv, json5, tiktoken, BeautifulSoup4, PyYAML


使用方式

配置环境（.env）：
DASHSCOPE_API_KEY
SERPER_API_KEYS
JINA_API_KEYS
AGENT_MODEL=qwen3.5-plus

启动服务：Bashuvicorn agent:app --reload
批量评测：Bashpython eval.py 0 100 my_results.jsonl
单题测试：通过 FastAPI 接口或直接调用 react_agent(question)


适用场景
需要极致准确率和可解释性的研究型 Agent
作为基础框架扩展自定义 Skills（skills.py 已准备好插件系统）