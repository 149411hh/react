import datetime
import json
import os
import re
import time
from typing import Optional, List, Dict

import json5
import tiktoken
from openai import AsyncOpenAI

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from tools_search import batch_search
from tools_visit import visit_pages


# ====================== Configuration ======================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5-plus")

MAX_ROUNDS = 100
TIMEOUT_SECONDS = 540          # 9 minutes
MAX_TOKENS_ESTIMATE = 500_000
MAX_TOOL_RESULT_CHARS = 15_000


# ====================== LLM Client ======================
_client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
)

# Initialize tiktoken for accurate token counting
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    print("[agent_loop] tiktoken loaded successfully")
except Exception as e:
    print(f"[agent_loop] Failed to load tiktoken: {e}")
    _tokenizer = None


class ResearchAgent:
    """Core Research Agent class with ReAct pattern."""

    def __init__(self):
        self.client = _client
        self.tokenizer = _tokenizer

    @staticmethod
    def _today_date() -> str:
        """Return current date as string."""
        return datetime.date.today().strftime("%Y-%m-%d")

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate the number of tokens in messages."""
        if self.tokenizer:
            try:
                return sum(len(self.tokenizer.encode(msg.get("content", ""))) for msg in messages)
            except Exception:
                pass

        # Fallback estimation
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 1.5)

    def _truncate_tool_result(self, result: str) -> str:
        """Truncate tool result if it's too long."""
        if len(result) > MAX_TOOL_RESULT_CHARS:
            return result[:MAX_TOOL_RESULT_CHARS] + "\n\n[... result truncated ...]"
        return result

    def _normalize_answer(self, answer: str) -> str:
        """Clean and normalize the final answer."""
        text = answer.strip()

        # Remove common prefixes
        prefixes = [
            "The answer is ", "the answer is ",
            "Answer: ", "answer: ",
            "答案是", "答案："
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove wrapping quotes
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
                text = text[1:-1].strip()
            elif (text[0] == '\u201c' and text[-1] == '\u201d') or (text[0] == '\u2018' and text[-1] == '\u2019'):
                text = text[1:-1].strip()
            elif (text[0] == '\u300c' and text[-1] == '\u300d'):
                text = text[1:-1].strip()

        return text

    def _extract_between(self, text: str, start_tag: str, end_tag: str) -> str:
        """Extract content between two tags."""
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return ""
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            return text[start_idx:].strip()
        return text[start_idx:end_idx].strip()

    def _extract_tool_call(self, content: str) -> Optional[Dict]:
        """Extract tool call from model output (supports <tool_call> and bare JSON)."""
        # Priority: <tool_call> tag
        if "<tool_call>" in content:
            tool_str = self._extract_between(content, "<tool_call>", "</tool_call>")
            if tool_str:
                try:
                    return json5.loads(tool_str)
                except:
                    pass

        # Fallback: bare JSON
        match = re.search(r'\{.*?"name":\s*"(search|visit)".*?\}', content, re.DOTALL)
        if match:
            try:
                return json5.loads(match.group(0))
            except:
                pass

        return None

    async def _call_llm(self, messages: List[Dict], temperature: float = 0.4, max_tokens: int = 8192) -> str:
        """Call the LLM and return the response content."""
        try:
            resp = await self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": True},
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", "") or ""
            content = msg.content or ""

            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n{content}".strip()

            return content.strip()

        except Exception as e:
            print(f"[agent] LLM call failed: {e}")
            return "I was unable to process this request due to an API error."

    async def _execute_tool(self, tool_name: str, tool_args: Dict) -> str:
        """Execute the requested tool and return the result."""
        try:
            if tool_name == "search":
                queries = tool_args.get("query", [])
                engines = tool_args.get("engine")
                if isinstance(queries, str):
                    queries = [queries]
                result = batch_search(queries, engines=engines)
                return self._truncate_tool_result(result)

            elif tool_name == "visit":
                urls = tool_args.get("url", [])
                goal = tool_args.get("goal", "Extract relevant information")
                if isinstance(urls, str):
                    urls = [urls]
                result = await visit_pages(urls, goal)
                return self._truncate_tool_result(result)

            return f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            return f"Tool execution error: {str(e)}"

    async def _force_answer(self, messages: List[Dict]) -> str:
        """Force the model to generate a final answer when limits are reached."""
        force_msg = {
            "role": "user",
            "content": "Based on all gathered information, provide the final answer now.\n"
                       "Use <think> to verify constraints, then output <answer>your answer</answer>"
        }
        messages.append(force_msg)

        content = await self._call_llm(messages, temperature=0.3)
        messages.append({"role": "assistant", "content": content})

        if "<answer>" in content:
            answer = self._extract_between(content, "<answer>", "</answer>")
            if answer:
                return self._normalize_answer(answer)

        return self._normalize_answer(content)

    async def run(self, question: str) -> str:
        """Main execution loop of the Research Agent."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + self._today_date()},
            {"role": "user", "content": USER_PROMPT_TEMPLATE + question},
        ]

        start_time = time.time()
        early_answer_blocked = False

        for round_idx in range(MAX_ROUNDS):
            # Timeout check
            if time.time() - start_time > TIMEOUT_SECONDS:
                print(f"[agent] Timeout reached, forcing answer")
                return await self._force_answer(messages)

            # Token limit check
            if self._estimate_tokens(messages) > MAX_TOKENS_ESTIMATE:
                print(f"[agent] Token limit reached, forcing answer")
                return await self._force_answer(messages)

            content = await self._call_llm(messages)
            messages.append({"role": "assistant", "content": content})

            print(f"[agent] Round {round_idx + 1} | Tokens ≈ {self._estimate_tokens(messages)}")

            # Check for final answer
            if "<answer>" in content:
                answer_text = self._extract_between(content, "<answer>", "</answer>") or \
                              content[content.find("<answer>") + 8:].strip()

                # Early answer blocking for better quality
                if answer_text and round_idx < 3 and not early_answer_blocked:
                    early_answer_blocked = True
                    print("[agent] Early answer detected, requesting verification")
                    messages.append({"role": "user", "content": "Please verify with one more source before finalizing."})
                    continue

                return self._normalize_answer(answer_text)

            # Extract and execute tool call
            tool_data = self._extract_tool_call(content)
            if tool_data:
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                print(f"[agent] Executing tool: {tool_name}")

                result = await self._execute_tool(tool_name, tool_args)
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>"
                })
            else:
                # Nudge the model when no tool or answer is provided
                messages.append({
                    "role": "user",
                    "content": "Please use <think> then make a tool call or provide the final <answer>."
                })

        # Max rounds reached
        print(f"[agent] Max rounds ({MAX_ROUNDS}) reached, forcing answer")
        return await self._force_answer(messages)


# ====================== Public Interface ======================
async def react_agent(question: str) -> str:
    """Public function to maintain backward compatibility."""
    agent = ResearchAgent()
    return await agent.run(question)
