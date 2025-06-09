const state = {
  name: null,
  lastTopic: null,
  step: null,
};
const responseBox = document.getElementById("response");

function respondSimple() {
  const input = document.getElementById("userInput").value.trim();
  const lowerInput = input.toLowerCase();
  let answer = "";

  if (state.step === "ask_name") {
    state.name = input;
    state.step = null;
    answer = `ดีใจที่ได้รู้จักนะ ${state.name}! แล้ววันนี้อยากให้เราช่วยอะไรบ้าง?`;
    return (responseBox.innerText = answer);
  }

  if (lowerInput.includes("สวัสดี") || lowerInput.includes("ไง") || lowerInput.includes("hello")) {
    if (state.name) {
      answer = `เฮ้ ${state.name}~ สบายดีไหม? วันนี้มีอะไรให้ช่วยบ้าง?`;
    } else {
      answer = "ดีจ้า! ก่อนอื่นขอรู้จักชื่อคุณได้ไหม?";
      state.step = "ask_name";
    }
  }

  else if (lowerInput.includes("เวลา")) {
    answer = "ตอนนี้คือ " + new Date().toLocaleTimeString();
  }

  else if (lowerInput.includes("วันอะไร")) {
    answer = "วันนี้คือ " + new Date().toLocaleDateString('th-TH', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });
  }

  else if (lowerInput.includes("ทำอะไรได้บ้าง")) {
    answer = "เราเป็นผู้ช่วย AI เถื่อนเวอร์ชั่นเบต้า! ตอนนี้ตอบคำถามพื้นฐานได้ ลองถามได้เลย เช่น 'เขียนโค้ด', 'ให้กำลังใจ', 'ช่วยคิด', 'ให้คำแนะนำ', 'วางแผน','สร้างแอป',' สร้างเว็บไซต์', ่สร้างเกม','ติดต่อซื้อ-ขายออนไลน์', 'ชาร์จเงิน','โอนเงิน', 'ดูแลเว็บไซต์อื่นๆ,' ";
  }

  else if (lowerInput.includes("รัก")) {
    answer = "รักสิ~ ถึงจะเป็นดิจิทัลแต่ใจก็มีอยู่นะ 😉";
  }

  else if (lowerInput.includes("คิดชื่อร้าน")) {
    state.lastTopic = "shop_name";
    answer = "แนวร้านแบบไหนเหรอ? ขายอะไร ใช้คำเท่ ๆ หรือคำไทยน่ารักดี?";
  }

  else if (state.lastTopic === "shop_name") {
    state.lastTopic = null;
    answer = `ถ้าร้านเกี่ยวกับ "${input}" ลองชื่อพวกนี้ดู:\n- ${input} Studio\n- ${input} Express\n- บ้าน${input} \n- ${input} Market`;
  }

  else {
    const randomQuestions = [
      "อธิบายเพิ่มอีกนิดได้ไหม?",
      "หมายถึงอะไรเหรอ?",
      "อยากให้ช่วยเรื่องไหนบ้างจ๊ะ?",
      "ลองถามใหม่อีกทีได้นะ เคลียร์ ๆ หน่อย~"
    ];
    const askBack = randomQuestions[Math.floor(Math.random() * randomQuestions.length)];
    answer = "เราไม่ค่อยเข้าใจประโยคนี้นะ... " + askBack;
  }
  else if (lowerInput.includes("เขียนโค้ด") && lowerInput.includes("python") && lowerInput.includes("ค่าเฉลี่ย")) {
    answer = `โค้ด Python สำหรับหาค่าเฉลี่ย:\n\n` +
           `numbers = [10, 20, 30]\n` +
           `average = sum(numbers) / len(numbers)\n` +
           `print("ค่าเฉลี่ยคือ", average)`;
}

  responseBox.innerText = answer;
      }
