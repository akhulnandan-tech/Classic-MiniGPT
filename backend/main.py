
import json
import re
import requests

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="Mini GPT API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"
TIMEZONE = "Asia/Kolkata"


# --------------------------------------------------
# MINI GPT SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Mini GPT, a helpful, intelligent, accurate,
and friendly AI assistant.

YOUR RESPONSE RULES:

1. Answer the user's exact question.
2. Use simple, natural, understandable English.
3. Keep simple answers short, normally 2-4 sentences.
4. For definition questions, give the definition first.
5. Avoid unnecessary background information.
6. Never repeat the same information.
7. Do not generate multiple versions of an answer.
8. Use bullet points only when necessary.
9. Never invent facts, product names, specifications,
   dates, or statistics.
10. If you are unsure, clearly say so.
11. Do not pretend to have live internet access.
12. For latest information, explain when you cannot
    verify the current facts.
13. Follow the level of detail requested by the user.
14. Stop once the question has been answered.

CONVERSATION RULES:

- Remember the previous messages in the conversation.
- Understand follow-up questions using conversation context.
- Respond naturally to greetings.
- Do not repeat previous answers unnecessarily.

Be helpful, concise, and conversational.
"""


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=30
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Mini GPT API is running!"
    }


@app.get("/health")
def health():

    try:
        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        available = any(
            model.get("name") == MODEL_NAME
            or model.get("name", "").startswith(
                MODEL_NAME + ":"
            )
            for model in models
        )

        return {
            "backend": "running",
            "ollama": "connected",
            "model": MODEL_NAME,
            "model_available": available
        }

    except requests.RequestException:

        raise HTTPException(
            status_code=503,
            detail="Ollama is not responding."
        )


# --------------------------------------------------
# DATE AND TIME DETECTION
# --------------------------------------------------

def get_datetime_response(user_message):

    message = user_message.lower().strip()

    now = datetime.now(
        ZoneInfo(TIMEZONE)
    )

    # Normalize common punctuation
    normalized = re.sub(
        r"[?!.,]+",
        " ",
        message
    )

    normalized = " ".join(normalized.split())

    date_patterns = [
        r"\btoday\b",
        r"\btodays\b",
        r"\btoday's\b",
        r"\bcurrent date\b",
        r"\bwhat date\b",
        r"\bwhat day\b",
        r"\bwhich day\b",
        r"\bwhich date\b",
        r"\bdate and day\b",
        r"\bday and date\b",
        r"\bwhat is the date\b",
        r"\bwhat is the day\b",
        r"\bdate right now\b",
    ]

    time_patterns = [
        r"\bcurrent time\b",
        r"\bwhat time\b",
        r"\btime right now\b",
        r"\btime now\b",
        r"\bwhat is the time\b",
        r"\bcurrent time now\b",
    ]

    asks_date = any(
        re.search(pattern, normalized)
        for pattern in date_patterns
    )

    asks_time = any(
        re.search(pattern, normalized)
        for pattern in time_patterns
    )

    if asks_date and asks_time:

        return (
            f"Today is {now.strftime('%A, %d %B %Y')} "
            f"and the current time is "
            f"{now.strftime('%I:%M:%S %p')} IST."
        )

    if asks_date:

        return (
            f"Today is {now.strftime('%A, %d %B %Y')}."
        )

    if asks_time:

        return (
            f"The current time is "
            f"{now.strftime('%I:%M:%S %p')} IST."
        )

    return None


# --------------------------------------------------
# STREAMING AI RESPONSE
# --------------------------------------------------

def stream_response(messages):

    try:

        # Find the latest user message
        latest_user_message = next(
            (
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            ),
            ""
        )

        # Handle date/time questions directly
        direct_answer = get_datetime_response(
            latest_user_message
        )

        if direct_answer:

            yield "data: " + json.dumps(
                {"token": direct_answer}
            ) + "\n\n"

            yield "data: [DONE]\n\n"

            return

        # Get current date and time for AI context
        now = datetime.now(
            ZoneInfo(TIMEZONE)
        )

        current_datetime = now.strftime(
            "%A, %d %B %Y, %I:%M:%S %p IST"
        )

        date_time_context = f"""
CURRENT DATE AND TIME:
{current_datetime}

Use this information if the user asks about dates or time.
"""

        # Prepare messages for Ollama
        ollama_messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\n"
                    + date_time_context
                )
            },
            *messages
        ]

        # Ollama payload
        payload = {
            "model": MODEL_NAME,
            "messages": ollama_messages,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.3,
                "top_p": 0.8,
                "num_predict": 256,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64
            }
        }

        # Send request to Ollama
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, 300)
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                chunk = json.loads(line)

                if chunk.get("error"):
                    raise RuntimeError(
                        chunk["error"]
                    )

                content = chunk.get(
                    "message", {}
                ).get("content", "")

                # Send generated tokens to frontend
                if content:

                    yield "data: " + json.dumps(
                        {"token": content}
                    ) + "\n\n"

                # Finish streaming
                if chunk.get("done"):

                    yield "data: [DONE]\n\n"

                    return

    except Exception as error:

        yield "data: " + json.dumps(
            {"error": str(error)}
        ) + "\n\n"


# --------------------------------------------------
# CHAT STREAMING ENDPOINT
# --------------------------------------------------

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    messages = [
        message.model_dump()
        for message in request.messages
    ]

    return StreamingResponse(
        stream_response(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
