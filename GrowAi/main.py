from brain import Brain

brain = Brain()

while True:
    question = input("ถามอะไรมาก็ได้ ('exit' เพื่อออก): ")
    if question.lower() == 'exit':
        break

    print(brain.understand(question))

    learn = input("อยากให้ฉันเรียนรู้คำใหม่มั้ย? (y/n): ")
    if learn.lower() == 'y':
        word = input("พิมพ์คำที่อยากสอน: ")
        meaning = input("ความหมายของคำนี้คือ: ")
        brain.learn(word, meaning)
        print("ได้เลย! ฉันจะจำคำว่า '{}': {}\n".format(word, meaning))
