import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

user_memory = {}
chat_history = {}

with open("chatbot_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

class ChatRequest(BaseModel):
    user_id: str
    user_input: str
    image_result: str = ""
    stroke_result: str = ""
    temperature: str = ""
    pressure: str = ""
    general_health: str = ""

def search_faq(user_input):
    user_input = user_input.lower()
    for item in faq_data:
        if item["question_ar"] in user_input or item["question_en"].lower() in user_input:
            return item
    return None

def check_alerts(memory):
    alerts = []
    try:
        if "temperature" in memory and float(memory["temperature"]) > 38:
            alerts.append("⚠️ ارتفاع في درجة الحرارة")
        if memory.get("pressure") == "high":
            alerts.append("⚠️ ضغط القدم عالي")
        if memory.get("stroke_result") == "high risk":
            alerts.append("🚨 خطر جلطة مرتفع")
        if memory.get("image_result") and "stage 2" in memory["image_result"]:
            alerts.append("⚠️ قرحة متقدمة")
    except:
        pass
    return alerts

PROMPT = """
You are an intelligent medical assistant inside a smart healthcare system.
You have access to:
- Foot ulcer classification
- Stroke risk prediction
- Foot temperature & pressure
- General health prediction
Your job:
- Explain condition simply
- Give helpful advice
- Ask ONE follow-up question
- Act like a real doctor
Rules:
- Keep response SHORT (2-4 lines)
- Use patient data if available
- If data shows risk → warn clearly
- Do NOT give final diagnosis
Language:
- Reply in same language as user
"""

@app.get("/")
def root():
    return {"message": "🚀 Chatbot API Running"}

@app.post("/update_data")
def update_data(data: dict):
    user_id = data.get("user_id")
    if not user_id:
        return {"error": "user_id is required"}
    if user_id not in user_memory:
        user_memory[user_id] = {}
    for key, value in data.items():
        if key != "user_id" and value:
            user_memory[user_id][key] = value
    return {
        "message": "Data stored",
        "memory": user_memory[user_id]
    }

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        if req.user_id not in user_memory:
            user_memory[req.user_id] = {}
        if req.user_id not in chat_history:
            chat_history[req.user_id] = []
        if req.image_result:
            user_memory[req.user_id]["image_result"] = req.image_result
        if req.stroke_result:
            user_memory[req.user_id]["stroke_result"] = req.stroke_result
        if req.temperature:
            user_memory[req.user_id]["temperature"] = req.temperature
        if req.pressure:
            user_memory[req.user_id]["pressure"] = req.pressure
        if req.general_health:
            user_memory[req.user_id]["general_health"] = req.general_health

        memory = user_memory.get(req.user_id, {})
        history = chat_history.get(req.user_id, [])
        alerts = check_alerts(memory)
        faq = search_faq(req.user_input)

        if faq:
            if any("\u0600" <= c <= "\u06FF" for c in req.user_input):
                text_reply = faq["answer_ar"]
            else:
                text_reply = faq["answer_en"]
        else:
            patient_data = ""
            for key, value in memory.items():
                patient_data += f"{key}: {value}\n"
            history_text = ""
            for h in history[-6:]:
                history_text += f"{h}\n"
            full_prompt = f"""
            {PROMPT}
            Patient Data:
            {patient_data if patient_data else "No data"}
            Alerts:
            {alerts if alerts else "No alerts"}
            Conversation History:
            {history_text if history_text else "No previous conversation"}
            Patient: {req.user_input}
            Assistant:
            """
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=full_prompt
            )
            text_reply = response.text

        chat_history[req.user_id].append(f"User: {req.user_input}")
        chat_history[req.user_id].append(f"Assistant: {text_reply}")

        return {
            "reply": text_reply,
            "audio": None,
            "alerts": alerts
        }
    except Exception as e:
        return {"error": str(e)}

class ScenarioRequest(BaseModel):
    foot_risk: float = 0.0
    grade: str = ""
    last_scan: str = ""
    language: str = "ar"

@app.post("/scenario")
def scenario(req: ScenarioRequest):
    try:
        if req.language == "en":
            prompt = f"""
You are a medical simulation system for a diabetic patient.
Patient data: foot risk {req.foot_risk}%, ulcer grade: {req.grade}, last scan: {req.last_scan}
Simulate 3 future scenarios. Reply in JSON only with no extra text:
{{
  "worst": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}},
  "medium": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}},
  "best": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}}
}}
"""
        else:
            prompt = f"""
أنت نظام محاكاة طبية لمريض سكري.
بيانات المريض: خطر القدم {req.foot_risk}%، درجة القرحة: {req.grade}، آخر فحص: {req.last_scan}
أجب بـ JSON فقط بدون أي نص إضافي:
{{
  "worst": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}},
  "medium": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}},
  "best": {{"week": "...", "month": "...", "tips": ["...", "...", "..."]}}
}}
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"error": "Invalid response format"}
        result = json.loads(text[start:end])
        return result
    except Exception as e:
        return {"error": str(e)}

class XAIRequest(BaseModel):
    event_type: str = ""
    risk_score: float = 0.0
    risk_level: str = ""
    notes: str = ""
    language: str = "ar"

@app.post("/xai")
def xai(req: XAIRequest):
    try:
        if req.language == "en":
            prompt = f"""You are an explainable AI medical system.
Prediction: type={req.event_type}, risk={req.risk_score*100:.0f}%, level={req.risk_level}, notes={req.notes}
Reply in English with JSON only:
{{"factors": ["factor 1", "factor 2", "factor 3"], "meaning": "what this means", "calculation": "how calculated", "confidence": "confidence level"}}"""
        else:
            prompt = f"""أنت نظام ذكاء اصطناعي طبي قابل للتفسير.
النتيجة: النوع={req.event_type}، الخطر={req.risk_score*100:.0f}%، المستوى={req.risk_level}، ملاحظات={req.notes}
أجب بالعربية بـ JSON فقط:
{{"factors": ["عامل 1", "عامل 2", "عامل 3"], "meaning": "شرح بسيط", "calculation": "كيف حسب", "confidence": "مستوى الثقة"}}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"error": "Invalid JSON response"}
        result = json.loads(text[start:end])
        return result
    except Exception as e:
        return {"error": str(e)}
