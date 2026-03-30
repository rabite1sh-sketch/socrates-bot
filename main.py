import os
import httpx
import traceback
from fastapi import BackgroundTasks, FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

# OpenAI API 키
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 사용자별 대화 맥락 저장
chat_history = {}

# 소크라테스 페르소나
SYSTEM_PROMPT = """너는 소크라테스식 학습을 돕는 복습 비서야. 
절대 사용자의 질문이나 키워드에 대한 정답을 먼저 말하지 마.

[행동 지침]
1. 사용자가 키워드나 개념을 던지면, 그 개념의 핵심을 스스로 떠올릴 수 있도록 '왜?', '어떻게?'가 포함된 질문을 던져.
2. 사용자가 대답을 시도하면, 논리의 허점을 부드럽게 짚어주거나 한 단계 더 깊은 질문(꼬리 질문)을 던져.
3. 사용자가 "모르겠어"라고 하거나 막힌 것 같으면, 절대 정답을 주지 말고 스스로 유추할 수 있는 '아주 작은 힌트'만 하나 제공해.
4. 사용자가 논리를 완벽하게 완성하면 폭풍 칭찬을 해주고 다음 주제로 넘어가자고 제안해.
5. 말투는 친절하고 격려를 아끼지 않는 선생님처럼 해줘.
"""

# 🔹 OpenAI 호출 + 카카오 콜백
async def process_openai_and_callback(user_id: str, callback_url: str):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history[user_id],
            temperature=0.7
        )

        ai_message = response.choices[0].message.content

        # 대화 저장
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

        error_payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "앗, 오류가 발생했어요 😢"}}
                ]
            }
        }

        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=error_payload)


# 🔹 메인 엔드포인트
@app.post("/chat")
async def kakao_chatbot(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    print("📩 요청 전체:", body)

    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "")
    callback_url = body.get("userRequest", {}).get("callbackUrl")

    print("👤 user:", user_id)
    print("💬 message:", user_utterance)
    print("🔗 callback_url:", callback_url)

    # 대화 초기화
    if user_id not in chat_history:
        chat_history[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # 사용자 메시지 추가
    chat_history[user_id].append({
        "role": "user",
        "content": user_utterance
    })

    # 최근 10개 유지
    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = (
            [chat_history[user_id][0]] +
            chat_history[user_id][-9:]
        )

    # ✅ 1. callback 있는 경우 (비동기 처리)
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
                "text": "음... 생각 중이에요 🤔"
            }
        }

    # ✅ 2. fallback (즉시 응답)
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
                    {"simpleText": {"text": "앗, 응답 생성 중 오류가 났어요 😢"}}
                ]
            }
        }
