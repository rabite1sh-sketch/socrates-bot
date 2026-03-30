import os
import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

# OpenAI API 키 설정
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 사용자별 대화 맥락 유지 저장소
chat_history = {}

# 소크라테스 페르소나 프롬프트
SYSTEM_PROMPT = """너는 소크라테스식 학습을 돕는 복습 비서야. 
절대 사용자의 질문이나 키워드에 대한 정답을 먼저 말하지 마.

[행동 지침]
1. 사용자가 키워드나 개념을 던지면, 그 개념의 핵심을 스스로 떠올릴 수 있도록 '왜?', '어떻게?'가 포함된 질문을 던져.
2. 사용자가 대답을 시도하면, 논리의 허점을 부드럽게 짚어주거나 한 단계 더 깊은 질문(꼬리 질문)을 던져.
3. 사용자가 "모르겠어"라고 하거나 막힌 것 같으면, 절대 정답을 주지 말고 스스로 유추할 수 있는 '아주 작은 힌트'만 하나 제공해.
4. 사용자가 논리를 완벽하게 완성하면 폭풍 칭찬을 해주고 다음 주제로 넘어가자고 제안해.
5. 말투는 친절하고 격려를 아끼지 않는 선생님처럼 해줘."""


async def process_openai_and_callback(user_id: str, callback_url: str):
    """
    백그라운드에서 OpenAI를 호출하고, 완료되면 카카오톡으로 콜백(답변)을 보내는 함수
    """
    try:
        # 1. OpenAI API 호출 (소크라테스 봇 응답 생성)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=chat_history[user_id],
            temperature=0.7 
        )
        
        ai_message = response.choices[0].message.content

        # 2. AI 응답을 대화 기록에 추가
        chat_history[user_id].append({"role": "assistant", "content": ai_message})

        # 3. 카카오 콜백 형식에 맞게 JSON 응답 구성
        callback_payload = {
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
        
        # 4. 카카오 서버로 답변(콜백) 전송
        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=callback_payload)

    except Exception as e:
        # 에러 발생 시에도 사용자에게 안내 메시지 전송
        error_message = f"앗, 비서가 잠깐 생각에 잠겼어요. 다시 한 번 말해주시겠어요? (에러: {str(e)})"
        error_payload = {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": error_message}}]}
        }
        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=error_payload)


@app.post("/chat")
async def kakao_chatbot(request: Request, background_tasks: BackgroundTasks):
    # 1. 카카오톡에서 보낸 JSON 데이터 파싱
    body = await request.json()
    
    # 카카오톡 사용자 고유 ID, 메시지, 그리고 콜백 URL 추출
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown_user")
    user_utterance = body.get("userRequest", {}).get("utterance", "")
    callback_url = body.get("userRequest", {}).get("callbackUrl")

    # 2. 사용자별 대화 기록 초기화
    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 3. 사용자의 현재 메시지를 대화 기록에 추가
    chat_history[user_id].append({"role": "user", "content": user_utterance})

    # 최근 10개의 대화만 유지하여 토큰 낭비 방지 (System prompt + 최근 9개)
    if len(chat_history[user_id]) > 10:
        chat_history[user_id] = [chat_history[user_id][0]] + chat_history[user_id][-9:]

    # 4. 콜백 URL이 있는 경우 (정상적인 카카오톡 요청) 백그라운드 작업 실행
    if callback_url:
        background_tasks.add_task(process_openai_and_callback, user_id, callback_url)

        # 카카오톡 서버에 즉시 응답 (5초 룰 통과용)
        return {
            "version": "2.0",
            "useCallback": True,
            "data": {
                "text": "음... 어떻게 설명하면 좋을지 생각 중이에요! 🤔"
            }
        }
    
    # (예외 처리) 콜백 URL이 없는 경우 - 로컬 테스트나 설정 오류 시
    else:
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "콜백 URL을 찾을 수 없습니다. 카카오 i 오픈빌더에서 '콜백 응답'을 켜주세요!"}}
                ]
            }
        }
