from flask import Flask, request, jsonify, render_template_string
from registry import get_model
import torch
from config import SYSTEM_PROMPT, USER_PROMPT, MODEL

app = Flask(__name__)

model = get_model(MODEL)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Q&A Interface</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        input { width: 100%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; margin-top: 10px; cursor: pointer; }
        #response { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; min-height: 50px; }
    </style>
</head>
<body>
    <h1>Medical Q&A Interface</h1>
    <input type="text" id="question" placeholder="Enter your medical question">
    <button onclick="ask()">Ask</button>
    <div id="response"></div>
    
    <script>
        async function ask() {
            const q = document.getElementById('question').value;
            const resp = document.getElementById('response');
            resp.textContent = 'Processing...';
            
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            
            const data = await res.json();
            resp.textContent = data.answer;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask():
    global model
    
    data = request.json
    question = data.get('question', '')
    
    if model is None:
        model = get_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    prompt_length = len(model.tokenizer.encode(prompt))  # Changed from model.tok
    
    with torch.no_grad():
        response = model.generate(prompt)
    
    tokens = model.tokenizer.encode(response)  # Changed from model.tok
    answer_tokens = tokens[prompt_length:]
    answer = model.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()  # Changed from model.tok
    
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    if "\n\n" in answer:
        answer = answer.split("\n\n")[0].strip()
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)