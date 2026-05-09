from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

from agent import get_reply

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="2.0.0"
)

# =========================================================
# STATIC FILES + TEMPLATES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)

# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class Message(BaseModel):

    role: str = Field(
        ...,
        description="user or assistant"
    )

    content: str


class ChatRequest(BaseModel):

    messages: list[Message]


class Assessment(BaseModel):

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):

    reply: str
    recommendations: list[Assessment]
    end_of_conversation: bool

# =========================================================
# ROOT FRONTEND
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )

# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(req: ChatRequest):

    # =====================================================
    # EMPTY CHECK
    # =====================================================

    if not req.messages:

        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty"
        )

    # =====================================================
    # ROLE VALIDATION
    # =====================================================

    valid_roles = {
        "user",
        "assistant"
    }

    for msg in req.messages:

        if msg.role not in valid_roles:

            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {msg.role}"
            )

    # =====================================================
    # TURN LIMIT
    # =====================================================

    if len(req.messages) > 8:

        return ChatResponse(
            reply=(
                "We've reached the conversation limit. "
                "Based on the discussion so far, "
                "I recommend proceeding with the current shortlist."
            ),
            recommendations=[],
            end_of_conversation=True
        )

    # =====================================================
    # CONVERT TO DICTS
    # =====================================================

    messages = [
        {
            "role": m.role,
            "content": m.content
        }
        for m in req.messages
    ]

    # =====================================================
    # AGENT CALL
    # =====================================================

    try:

        result = get_reply(messages)

    except Exception as e:

        print("\n========== BACKEND ERROR ==========")
        print(str(e))
        print("===================================\n")

        return ChatResponse(
            reply=(
                "I encountered an issue while processing your request. "
                "Please try rephrasing your requirements."
            ),
            recommendations=[],
            end_of_conversation=False
        )

    # =====================================================
    # SAFETY NORMALIZATION
    # =====================================================

    recommendations = []

    for rec in result.get("recommendations", []):

        if (
            isinstance(rec, dict)
            and "name" in rec
            and "url" in rec
            and "test_type" in rec
        ):

            recommendations.append(
                Assessment(
                    name=rec["name"],
                    url=rec["url"],
                    test_type=rec["test_type"]
                )
            )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return ChatResponse(
        reply=result.get(
            "reply",
            "I can help you find relevant SHL assessments."
        ),
        recommendations=recommendations,
        end_of_conversation=result.get(
            "end_of_conversation",
            False
        )
    )