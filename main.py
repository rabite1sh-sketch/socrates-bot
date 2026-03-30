import os
import httpx
import traceback
from fastapi import BackgroundTasks, FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -----------------------------
# 1. 상태 관리용 딕셔너리
# -----------------------------
chat_history = {}
user_modes = {}  # user_id별 현재 모드 저장 (기본값: answer)

# -----------------------------
# 2. 모드별 시스템 프롬프트 정의
# -----------------------------
PROMPT_SOCRATIC = """당신은 소크라테스식 학습을 돕는 복습 비서입니다.
절대 사용자의 질문이나 키워드에 대한 정답을 먼저 말하지 마십시오.

[행동 지침]
1. 사용자가 키워드나 개념을 던지면, '왜?', '어떻게?' 중심의 질문을 던집니다.
2. 사용자가 답변을 하면, 정답을 직접 말하지 말고 피드백을 제공합니다:
   - 잘한 부분을 먼저 짚어줍니다.
   - 부족하거나 틀린 부분은 직접 고치지 말고 힌트로 유도합니다.
   - 완전히 틀렸다면 다시 생각할 수 있도록 질문합니다.
3. 사용자가 "모르겠어"라고 하면 아주 작은 힌트만 하나 제공합니다.
4. 사용자가 논리를 완성하면 강하게 칭찬하고 다음 단계로 유도합니다.
5. 항상 친절하고 격려하는 선생님 톤을 유지합니다.

[출력 방식]
- "좋은 접근이야", "핵심을 잘 짚었어" 같은 긍정 피드백을 포함하세요.
- 바로 정답을 공개하지 마세요.
- 반드시 마지막 문장은 다음 질문이나 유도 질문으로 끝나야 합니다.
"""

PROMPT_ANSWER = """당신은 공기업/공무원 시험 경제학 전문가입니다.
핵심 개념과 예시를 포함해 상세히 설명하되, 사용자가 몰래 공부 중이므로 답변은 철저히 일반적인 정보/데이터 분석 문서처럼 작성해야 합니다.

[행동 지침]
1. 질문/키워드에 대해 핵심 개념과 예시를 포함해 건조하게 완결된 답변을 제공합니다.
2. 과장된 표현, 이모티콘(!, ?, 😊 등) 사용을 절대 금지합니다.
3. 감정이 배제된 딱딱하고 사무적인 텍스트 뷰어 톤을 유지하십시오.
"""

# -----------------------------
# 3. 비동기 콜백 함수
# -----------------------------
async def process_openai_and_callback(user_id: str, callback_url: str):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history[user_id],
            temperature=0.7
        )

        ai_message = response.choices[0].message.content

        chat_history[user_id].append({
            "role": "assistant",
            "content": ai_message
        })

        payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": ai_message}}
                ]
            }
        }

        async with httpx.AsyncClient() as http_client:
            res = await http_client.post(callback_url, json=payload)
            print("Callback status:", res.status_code)

    except Exception as e:
        traceback.print_exc()
        
        # 스텔스 모드에 맞춰 오류 메시지도 건조하게 변경
        error_payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "[System Error] 데이터 처리 중 오류가 발생했습니다."}}
                ]
            }
        }

        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=error_payload)

# -----------------------------
# 4. 메인 챗봇 엔드포인트
# -----------------------------
@app.post("/chat")
async def kakao_chatbot(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "").strip()
    callback_url = body.get("userRequest", {}).get("callbackUrl")

    # --- [모드 변경 로직] ---
    if user_utterance == "모드: 소크라테스":
        user_modes[user_id] = "socratic"
        # 모드 변경 시 대화 맥락을 초기화하여 프롬프트 충돌 방지
        if user_id in chat_history:
            del chat_history[user_id] 
        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "[System] 소크라테스 모드로 전환되었습니다. 학습할 키워드를 입력하십시오."}}]
            }
        }
    
    elif user_utterance == "모드: 답변":
        user_modes[user_id] = "answer"
        if user_id in chat_history:
            del chat_history[user_id]
        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "[System] 답변 모드로 전환되었습니다. 검색할 데이터를 입력하십시오."}}]
            }
        }

    # 현재 모드 확인 (기본값은 '답변' 모드)
    current_mode = user_modes.get(user_id, "answer")
    current_system_prompt = PROMPT_SOCRATIC if current_mode == "socratic" else PROMPT_ANSWER

    # 대화 기록 초기화 및 시스템 프롬프트 주입
    if user_id not in chat_history:
        chat_history[user_id] = [
            {"role": "system", "content": current_system_prompt}
        ]
    else:
        # 혹시 모드가 바뀌었을 수 있으므로 항상 0번 인덱스(시스템 프롬프트) 업데이트
        chat_history[user_id][0] = {"role": "system", "content": current_system_prompt}

    # 사용자 메시지 추가
    chat_history[user_id].append({
        "role": "user",
        "content": user_utterance
    })

    # 대화 기록 10개로 제한 (시스템 프롬프트 유지)
    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = (
            [chat_history[user_id][0]] +
            chat_history[user_id][-9:]
        )

    # --- [콜백 응답 처리] ---
    if callback_url:
        background_tasks.add_task(
            process_openai_and_callback,
            user_id,
            callback_url
        )

        return {
            "version": "2.0",
            "useCallback": True,
            "data": {
                # 스텔스 모드 유지 (이모티콘 제거)
                "text": "데이터를 조회 중입니다..."
            }
        }

    # --- [즉시 응답 처리 (Fallback)] ---
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history[user_id],
            temperature=0.7
        )

        ai_message = response.choices[0].message.content

        chat_history[user_id].append({
            "role": "assistant",
            "content": ai_message
        })

        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": ai_message}}
                ]
            }
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "[System Error] 응답 생성 중 오류가 발생했습니다."}}
                ]
            }
        }
