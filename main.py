# STEP 1: ดึงโปรเจกต์จาก GitHub 
!git clone
https://github.com/Sing9999/Mengsing-The-Digital.git
# STEP 1: ติดตั้งไลบรารีที่จำเป็น
!pip install python-telegram-bot==20.7 transformers pyngrok
transformers
# STEP 2: Import และตั้งค่าตัวแปร
import os
from pyngrok import ngrok
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# STEP 3: Telegram Token (ใส่ของน้องไว้ตรงนี้)
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"  # 👈 แก้เป็นของจริง

# STEP 4: Ngrok Token (สมัครที่ https://dashboard.ngrok.com/signup แล้วเอา Token มาใส่)
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # 👈 แก้เป็นของจริง

# STEP 5: สร้าง MemoryBot
class MemoryBot:
    def __init__(self):
        self.memory_file = "user_memory.json"
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def remember(self, user_id, key, value):
        if str(user_id) not in self.memory:
            self.memory[str(user_id)] = {}
        self.memory[str(user_id)][key] = value
        self.save_memory()

    def recall(self, user_id):
        return self.memory.get(str(user_id), {})

# STEP 6: เตรียมโมเดลภาษาไทย
model_name = "flax-community/gpt2-thai"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

HISTORY_FILE = "conversation_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def get_response(user_input, history):
    context = "\n".join([f"User: {h['input']} Bot: {h['response']}" for h in history[-3:]])
    prompt = f"{context}\nUser: {user_input}\nBot:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Bot:")[-1].strip()
    return response

# STEP 7: สร้างฟังก์ชันแชท
brain = MemoryBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(f"สวัสดี {user.first_name}! พร้อมคุยแล้วครับ 🧠")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    if "จำได้ไหม" in text:
        memory = brain.recall(user_id)
        if memory:
            lines = [f"- {k}: {v}" for k, v in memory.items()]
            await update.message.reply_text("คุณเคยบอกไว้ว่า:\n" + "\n".join(lines))
        else:
            await update.message.reply_text("ยังไม่มีข้อมูลเลย ลองสอนฉันสิ")

    elif "=" in text:
        try:
            key, value = [t.strip() for t in text.split("=", 1)]
            brain.remember(user_id, key, value)
            await update.message.reply_text(f"รับทราบ! จำว่า '{key}' คือ '{value}' แล้ว")
        except:
            await update.message.reply_text("พิมพ์ไม่ถูกนะ ลองใหม่แบบนี้: ชื่อ = คำอธิบาย")

    else:
        history = load_history()
        response = get_response(text, history)
        await update.message.reply_text(f"AI: {response}")
        history.append({"input": text, "response": response})
        save_history(history)

# STEP 8: สร้าง Webhook ด้วย ngrok
os.system(f"ngrok config add-authtoken {NGROK_AUTH_TOKEN}")
public_url = ngrok.connect(8443, "http")
print(f"🌐 Public URL สำหรับ Webhook: {public_url}")

# STEP 9: เริ่มรันบอท
from telegram.ext import Defaults
app = ApplicationBuilder().token(TELEGRAM_TOKEN).webhook_url(public_url).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

app.run_webhook(listen="0.0.0.0", port=8443, webhook_url=public_url)
