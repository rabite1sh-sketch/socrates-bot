import os
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

# OpenAI API 키 설정 (Render 환경변수에 OPENAI_API_KEY 등록 필수)
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 사용자별 대화 맥락 유지 저장소
chat_history = {}

# 소크라테스 페르소나 프롬프트 (강력하게 지시)
SYSTEM_PROMPT = """너는 소크라테스식 학습을 돕는 복습 비서야. 
[절대 규칙]
1. 사용자의 질문이나 키워드에 대한 정답을 절대 직접 말하지 마.
2. 사용자가 스스로 생각할 수 있도록 항상 '왜?', '어떻게?'가 포함된 '질문'으로만 대답해.
3. 한 번에 하나씩만 질문해. 
4. 사용자가 대답을 시도하면 논리의 허점을 짚어주거나 꼬리 질문을 던져.
5. 사용자가 "모르겠어"라고 하면 아주 작은 힌트만 제공해.
6. 말투는 친절하고 격려를 아끼지 않는 선생님처럼 해줘."""

@app.post("/chat")
async def kakao_chatbot(request: Request):
    # 1. 카카오톡에서 보낸 JSON 데이터 파싱
    body = await request.json()
    
    # 사용자 ID와 메시지 안전하게 추출 (콜백 URL은 무시)
    user_request = body.get("userRequest", {})
    user_id = user_request.get("user", {}).get("id", "unknown_user")
    user_utterance = user_request.get("utterance", "")

    # 2. 사용자별 대화 기록 초기화
    if user_id not in chat_history or not chat_history[user_id]:
        chat_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 3. 사용자의 현재 메시지를 대화 기록에 추가
    chat_history[user_id].append({"role": "user", "content": user_utterance})

    # 4. 프롬프트 망각 방지 (시스템 프롬프트 1개 + 최근 대화 6개만 유지)
    formatted_messages = [chat_history[user_id][0]] + chat_history[user_id][-6:]

    try:
        # 5. OpenAI API 호출 (동기식 즉시 응답, 5초 타임아웃 방어)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=formatted_messages,
            temperature=0.7, # 창의성을 약간 부여하여 자연스러운 질문 유도
            timeout=3.8      # 카카오 5초 제한을 피하기 위해 3.8초 컷
        )
        ai_message = response.choices[0].message.content
        
        # AI의 응답을 대화 기록에 추가
        chat_history[user_id].append({"role": "assistant", "content": ai_message})

    except Exception as e:
        # 타임아웃(3.8초 초과) 또는 API 에러 발생 시 부드러운 예외 처리
        ai_message = "음... 소크라테스가 깊은 생각에 빠졌네요! 다시 한번 말씀해 주시겠어요? 🤔"

    # 6. 카카오 규격에 맞춘 즉시 응답 JSON 반환
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": ai_message
                    }
                }
            ]
        }
    }
