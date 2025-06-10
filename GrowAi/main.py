🧐 AI Chatbot พร้อมหน่วยความจำ (Memory) และเชื่อม Telegram + ภาษาไทยด้วย HuggingFace 🧠🇳🇽

from telegram import Update, ForceReply from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters from transformers import AutoTokenizer, AutoModelForCausalLM import json import os

--------------------------- Memory Class ----------------------------

class MemoryBot: def init(self): self.memory_file = "user_memory.json" self.memory = self.load_memory()

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

--------------------------- AI ภาษาไทยด้วย Transformers ----------------------------

model_name = "flax-community/gpt2-thai" tokenizer = AutoTokenizer.from_pretrained(model_name) model = AutoModelForCausalLM.from_pretrained(model_name)

HISTORY_FILE = "conversation_history.json"

def load_history(): if os.path.exists(HISTORY_FILE): with open(HISTORY_FILE, 'r', encoding='utf-8') as f: return json.load(f) return []

def save_history(history): with open(HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(history, f, ensure_ascii=False, indent=4)

def get_response(user_input, history): context = "\n".join([f"User: {h['input']} Bot: {h['response']}" for h in history[-3:]]) prompt = f"{context}\nUser: {user_input}\nBot:"

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

--------------------------- Telegram Logic ----------------------------

brain = MemoryBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: user = update.effective_user await update.message.reply_text(f"สวัสดี {user.first_name}! ผมคือแชทบอทอัจฉริยะพร้อมหน่วยความจำ 🧠\nพิมพ์อะไรมาคุยได้เลยนะครับ!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): text = update.message.text.strip() user_id = update.effective_user.id

if "จำได้ไหม" in text:
    memory = brain.recall(user_id)
    if memory:
        lines = [f"- {k}: {v}" for k, v in memory.items()]
        await update.message.reply_text("นี่คือสิ่งที่คุณเคยบอกฉันไว้:\n" + "\n".join(lines))
    else:
        await update.message.reply_text("ผมยังไม่รู้อะไรเลยนะ ลองสอนผมดูสิ")

elif "=" in text:
    try:
        key, value = [t.strip() for t in text.split("=", 1)]
        brain.remember(user_id, key, value)
        await update.message.reply_text(f"โอเค! ผมจะจำว่า '{key}' หมายถึง '{value}'")
    except:
        await update.message.reply_text("รูปแบบไม่ถูกต้อนะครับ ลองพิมพ์แบบ: หัวข้อ = คำอธิบาย")

else:
    history = load_history()
    response = get_response(text, history)
    await update.message.reply_text(f"AI: {response}")
    history.append({"input": text, "response": response})
    save_history(history)

--------------------------- Run Telegram Bot ----------------------------

TOKEN = "8094720160:AAEqtK9y_DBedwcnPD8vdkpiWaOwVLUMucg"

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start)) app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if name == "main": print("🤖 เริ่มแชทบอท Telegram + ภาษาไทย...") app.run_polling()

