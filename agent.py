"""
FastAPI service for the Research Agent.

Provides a clean REST API endpoint to interact with the Research Agent.
Supports both normal JSON response and Server-Sent Events (SSE) streaming.
"""

import asyncio
import json
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

# Load environment variables
load_dotenv()

from agent_loop import react_agent

app = FastAPI(title="Research Agent API", version="1.0.0")


class QueryRequest(BaseModel):
    """Request model for agent query."""
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "Where is the capital of France?"}
        }
    )

    question: str
    chat_history: Optional[list] = None


class QueryResponse(BaseModel):
    """Response model."""
    answer: str


@app.post("/", response_model=QueryResponse)
async def query(req: QueryRequest, raw_request: Request):
    """
    Main endpoint for the Research Agent.

    Supports two response modes:
    - Normal JSON response (default)
    - Server-Sent Events (SSE) when Accept: text/event-stream is set
    """
    accept = raw_request.headers.get("accept", "")

    # SSE Streaming Mode
    if "text/event-stream" in accept:
        async def sse_response():
            try:
                # Run agent in background
                agent_task = asyncio.create_task(react_agent(req.question))

                # Send heartbeats to keep connection alive
                while not agent_task.done():
                    yield ": heartbeat\n\n"
                    try:
                        await asyncio.wait_for(asyncio.shield(agent_task), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue

                answer = await agent_task
            except Exception as e:
                print(f"[API] SSE error: {e}")
                answer = f"Error: {e}"

            data = json.dumps({"answer": answer}, ensure_ascii=False)
            yield f"event: Message\ndata: {data}\n\n"

        return StreamingResponse(sse_response(), media_type="text/event-stream")

    # Normal JSON Mode
    try:
        answer = await react_agent(req.question)
    except Exception as e:
        print(f"[API] Error: {e}")
        answer = f"Error: {e}"

    return JSONResponse(content={"answer": answer})


# Optional: Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Research Agent API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
