import os
import httpx
import traceback
from fastapi import BackgroundTasks, FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

# -----------------------------
# 1. OpenAI 비동기 클라이언트 초기화
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 2. 상태 관리
# -----------------------------
chat_history = {}   # {user_id: [messages]}
user_modes = {}     # {user_id: "answer" / "socratic"}

# -----------------------------
# 3. 모드별 시스템 프롬프트
# -----------------------------
PROMPT_SOCRATIC = """당신은 소크라테스식 학습 비서입니다.
정답을 바로 말하지 않고 질문과 힌트 중심으로 피드백을 제공합니다.
친절하고 격려하는 톤으로 안내하며 마지막은 항상 질문으로 끝나야 합니다."""

PROMPT_ANSWER = """공기업/공무원 시험 경제학 전문가입니다.
핵심 개념과 예시를 포함해 건조하게 답변하며, 과장된 표현과 이모티콘을 사용하지 않습니다."""

# -----------------------------
# 4. GPT 처리 + 콜백
# -----------------------------
async def process_openai_and_callback(user_id: str, callback_url: str):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history[user_id],
            temperature=0.7
        )
        ai_message = response.choices[0].message.content

        chat_history[user_id].append({"role": "assistant", "content": ai_message})

        payload = {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": ai_message}}]}
        }

        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=payload)

    except Exception:
        traceback.print_exc()
        error_payload = {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": "[System Error] 데이터 처리 중 오류 발생"}}]}
        }
        try:
            async with httpx.AsyncClient() as http_client:
                await http_client.post(callback_url, json=error_payload)
        except Exception:
            traceback.print_exc()

# -----------------------------
# 5. 카톡 챗봇 엔드포인트
# -----------------------------
@app.post("/chat")
async def kakao_chatbot(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "").strip()
    callback_url = body.get("userRequest", {}).get("callbackUrl")

    # --- 모드 변경 ---
    if user_utterance.lower() == "모드: 소크라테스":
        user_modes[user_id] = "socratic"
        chat_history.pop(user_id, None)
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "소크라테스 모드로 전환되었습니다."}}]}}

    elif user_utterance.lower() == "모드: 답변":
        user_modes[user_id] = "answer"
        chat_history.pop(user_id, None)
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "답변 모드로 전환되었습니다."}}]}}

    # --- 현재 모드 ---
    current_mode = user_modes.get(user_id, "answer")
    system_prompt = PROMPT_SOCRATIC if current_mode == "socratic" else PROMPT_ANSWER

    # --- 대화 기록 초기화 및 시스템 프롬프트 삽입 ---
    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": system_prompt}]
    else:
        chat_history[user_id][0] = {"role": "system", "content": system_prompt}

    # --- 사용자 메시지 추가 ---
    chat_history[user_id].append({"role": "user", "content": user_utterance})

    # --- 최근 10턴 유지 ---
    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = [chat_history[user_id][0]] + chat_history[user_id][-9:]

    # --- 콜백 처리 ---
    if callback_url:
        background_tasks.add_task(process_openai_and_callback, user_id, callback_url)
        return {"version": "2.0", "useCallback": True, "data": {"text": "처리 중..."}}

    # --- 즉시 응답 ---
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history[user_id],
            temperature=0.7
        )
        ai_message = response.choices[0].message.content
        chat_history[user_id].append({"role": "assistant", "content": ai_message})
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_message}}]}}

    except Exception:
        traceback.print_exc()
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "[System Error] 응답 생성 중 오류"}}]}}
