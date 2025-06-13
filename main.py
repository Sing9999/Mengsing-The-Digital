from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# โหลดโมเดลและ tokenizer
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# โหลดชุดข้อมูล (เช่น SQuAD หรือชุดข้อมูลที่คุณสร้างเอง)
dataset = load_dataset("squad")  # หรือชุดข้อมูลที่กำหนดเองในรูปแบบ JSON/CSV

# เตรียมข้อมูลสำหรับฝึก
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = examples["answers"]
    
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)
        
        # หาตำแหน่งเริ่มต้นและสิ้นสุดของบริบท
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1
        
        start_positions.append(context_start)
        end_positions.append(context_end)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# ตั้งค่าการฝึก
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# เริ่มฝึก
trainer.train()

# บันทึกโมเดล
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
