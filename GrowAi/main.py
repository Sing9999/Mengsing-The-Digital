# main_colab_gradio.py
from brain import Brain
import gradio as gr

brain = Brain()

def chat_with_brain(question, teach=False, word=None, meaning=None):
    response = brain.understand(question)
    learn_msg = ""
    if teach and word and meaning:
        brain.learn(word, meaning)
        learn_msg = f"\n\n🧠 ฉันจะจำคำว่า '{word}': {meaning}"
    return response + learn_msg

interface = gr.Interface(
    fn=chat_with_brain,
    inputs=[
        gr.Textbox(label="ถามอะไรมาก็ได้"),
        gr.Checkbox(label="อยากสอนคำใหม่ไหม?"),
        gr.Textbox(label="คำที่อยากสอน", optional=True),
        gr.Textbox(label="ความหมายของคำ", optional=True),
    ],
    outputs="text",
    title="Memory AI ของ MengSing",
    description="AI เรียนรู้คำจากคุณได้ ✨"
)

interface.launch(share=True)
