import os
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chat_history = {}

@app.post("/chat")
async def kakao_chatbot(request: Request):
    body = await request.json()
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown")
    user_utterance = body.get("userRequest", {}).get("utterance", "")

    # 1. 기록 초기화 (기존 로직)
    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": "너는 소크라테스 비서야."}]
    chat_history[user_id].append({"role": "user", "content": user_utterance})

    try:
        # 2. 핵심: 4초 안에 답 안 오면 에러로 던지기 (카카오 5초 제한 방어)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history[user_id],
            timeout=3.5  # 3.5초 안에 답변 안 오면 끊어버림
        )
        ai_message = response.choices[0].message.content
        
    except Exception as e:
        # 3. 시간이 초과되거나 에러 나면 '폴백' 대신 이 메시지를 전송
        ai_message = "음... 생각이 길어지네요! 다시 한번 말씀해 주시겠어요? 🤔"

    # 4. '콜백' 설정 없이도 작동하는 표준 응답 형식
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": ai_message}}]
        }
    }
