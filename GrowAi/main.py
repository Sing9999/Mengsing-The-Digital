from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# เริ่มต้นโมเดลและ tokenizer (ใช้โมเดลที่รองรับภาษาไทย)
model_name = "airesearch/wangchanberta-base-att-spm-uncased"  # โมเดลภาษาไทย
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ไฟล์สำหรับเก็บประวัติการสนทนา
HISTORY_FILE = "conversation_history.json"

def load_history():
    """โหลดประวัติการสนทนาจากไฟล์"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    """บันทึกประวัติการสนทนา"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def get_response(user_input, history):
    """สร้างการตอบสนองจากโมเดล"""
    # เพิ่มบริบทจากประวัติ (ถ้ามี)
    context = "\n".join([f"User: {h['input']} Bot: {h['response']}" for h in history[-3:]])  # ใช้ 3 การสนทนาล่าสุด
    prompt = f"{context}\nUser: {user_input}\nBot:"

    # เข้ารหัสอินพุต
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # สร้างการตอบสนอง
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # ถอดรหัสการตอบสนอง
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Bot:")[-1].strip()  # ดึงเฉพาะส่วนตอบกลับ
    
    return response

def main():
    print("สวัสดี! ฉัน Mengsing ที่เข้าใจภาษาไทย พิมพ์คำถามหรือคำศัพท์มาได้เลย (พิมพ์ 'ออก' เพื่อหยุด)")
    history = load_history()

    while True:
        user_input = input("คุณ: ")
        
        if user_input.lower() == "ออก":
            print("บายบาย!")
            break
        
        # สร้างการตอบสนอง
        response = get_response(user_input, history)
        print(f"AI: {response}")
        
        # อัปเดตประวัติ
        history.append({"input": user_input, "response": response})
        save_history(history)

if __name__ == "__main__":
    main()
