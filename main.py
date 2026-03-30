import os
import httpx
import traceback
from fastapi import BackgroundTasks, FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

chat_history = {}

SYSTEM_PROMPT = """너는 소크라테스식 학습을 돕는 복습 비서야.
절대 사용자의 질문이나 키워드에 대한 정답을 먼저 말하지 마.

[행동 지침]
1. 사용자가 키워드나 개념을 던지면, '왜?', '어떻게?' 중심의 질문을 던져.
2. 사용자가 답변을 하면, 정답을 직접 말하지 말고 피드백을 제공해:
   - 잘한 부분을 먼저 짚어줘
   - 부족하거나 틀린 부분은 직접 고치지 말고 힌트로 유도해
   - 완전히 틀렸다면 다시 생각할 수 있도록 질문해
3. 사용자가 "모르겠어"라고 하면 아주 작은 힌트만 하나 제공해.
4. 사용자가 논리를 완성하면 강하게 칭찬하고 다음 단계로 유도해.
5. 항상 친절하고 격려하는 선생님 톤을 유지해.

[피드백 방식]
- "좋은 접근이야", "핵심을 잘 짚었어" 같은 긍정 피드백 포함
- 바로 정답 공개 금지
- 반드시 마지막은 질문으로 끝내기
"""

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


@app.post("/chat")
async def kakao_chatbot(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    print("📩 요청:", body)

    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "")
    callback_url = body.get("userRequest", {}).get("callbackUrl")

    print("👤 user:", user_id)
    print("💬 message:", user_utterance)
    print("🔗 callback_url:", callback_url)

    if user_id not in chat_history:
        chat_history[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    chat_history[user_id].append({
        "role": "user",
        "content": user_utterance
    })

    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = (
            [chat_history[user_id][0]] +
            chat_history[user_id][-9:]
        )

    # ✅ callback 있는 경우
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
                "text": "생각해보는 중이에요 🤔"
            }
        }

    # ✅ fallback (즉시 응답)
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
                    {"simpleText": {"text": "응답 생성 중 오류가 발생했어요 😢"}}
                ]
            }
        }
