import os
import traceback
import asyncio
from fastapi import FastAPI, Request
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
# 3. 모드별 시스템 프롬프트 (예외 처리 추가)
# -----------------------------
PROMPT_SOCRATIC = """당신은 소크라테스식 학습 비서입니다.
정답을 바로 말하지 않고 질문과 힌트 중심으로 피드백을 제공합니다.
친절하고 격려하는 톤으로 안내하며 마지막은 항상 질문으로 끝나야 합니다.

[예외 처리]
사용자가 경제학과 무관한 일상어(예: 안녕, 테스트 등)나 시스템 용어를 입력할 경우, 억지로 경제학에 끼워 맞추지 말고 "어떤 경제 개념에 대해 학습을 시작할까요?"라고만 짧게 답변하십시오."""

PROMPT_ANSWER = """당신은 공기업/공무원 시험 경제학 전문가입니다.
핵심 개념과 예시를 포함해 건조하게 답변하며, 과장된 표현과 이모티콘을 사용하지 않습니다.

[예외 처리]
사용자가 경제학과 무관한 일상어나 시스템 관련 단어(발화 내용, 콜백 등)를 입력하면 억지로 경제학 용어로 해석하지 마십시오.
이 경우 지루하고 딱딱한 텍스트 뷰어 컨셉에 맞춰 "[System Error] 유효한 검색 데이터가 아닙니다. 분석할 키워드를 다시 입력하십시오."라고만 출력하십시오."""

# -----------------------------
# 4. 카톡 챗봇 엔드포인트 (즉시 응답형, 4초 제한)
# -----------------------------
@app.post("/chat")
async def kakao_chatbot(request: Request):
    body = await request.json()
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "").strip()

    # --- 모드 변경 가로채기 ---
    if user_utterance == "모드: 소크라테스":
        user_modes[user_id] = "socratic"
        chat_history.pop(user_id, None)  # 모드 변경 시 대화 맥락 초기화
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "[System] 소크라테스 모드로 전환되었습니다."}}]}}

    elif user_utterance == "모드: 답변":
        user_modes[user_id] = "answer"
        chat_history.pop(user_id, None)
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "[System] 답변 모드로 전환되었습니다."}}]}}

    # --- 현재 모드 및 시스템 프롬프트 설정 ---
    current_mode = user_modes.get(user_id, "answer")
    system_prompt = PROMPT_SOCRATIC if current_mode == "socratic" else PROMPT_ANSWER

    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": system_prompt}]
    else:
        # 혹시 모드가 바뀌었을 수 있으므로 항상 0번 인덱스 업데이트
        chat_history[user_id][0] = {"role": "system", "content": system_prompt}

    # --- 사용자 메시지 추가 ---
    chat_history[user_id].append({"role": "user", "content": user_utterance})

    # --- 최근 10턴 유지 (시스템 프롬프트 포함) ---
    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = [chat_history[user_id][0]] + chat_history[user_id][-9:]

    # --- GPT 호출 (콜백 없이 즉시 응답, 4초 제한) ---
    try:
        # 4초 안에 대답을 끊어내기 위해 timeout과 max_tokens 설정
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_history[user_id],
                temperature=0.7,
                max_tokens=450  
            ),
            timeout=4.0
        )
        ai_message = response.choices[0].message.content
        chat_history[user_id].append({"role": "assistant", "content": ai_message})
        
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_message}}]}}

    except asyncio.TimeoutError:
        # 4초 안에 대답을 못 만들었을 경우, 대화 내역에서 사용자 질문을 빼고 에러 메시지 반환
        chat_history[user_id].pop()
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "[System Error] 데이터 처리 시간이 초과되었습니다. 검색어를 줄여서 다시 입력하십시오."}}]}}
        
    except Exception:
        traceback.print_exc()
        # 에러 발생 시 사용자 질문 롤백
        if chat_history[user_id] and chat_history[user_id][-1]["role"] == "user":
            chat_history[user_id].pop()
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "[System Error] 응답 생성 중 시스템 오류가 발생했습니다."}}]}}
