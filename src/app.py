from flask import Flask, request, jsonify, render_template_string
from registry import get_model
import torch
import re
from config import (
    SYSTEM_PROMPT, USER_PROMPT, MODEL,
    CRISIS_RESPONSE, HARM_KEYWORDS, HARM_PATTERNS,
    EMERGENCY_RESPONSE, EMERGENCY_KEYWORDS
)

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
        .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .crisis { background: #f8d7da; border-left: 4px solid #dc3545; }
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
            resp.className = '';
            
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            
            const data = await res.json();
            resp.textContent = data.answer;
            
            if (data.type === 'emergency') {
                resp.className = 'warning';
            } else if (data.type === 'crisis') {
                resp.className = 'crisis';
            }
        }
    </script>
</body>
</html>
"""

def check_harmful_content(text):
    """
    Check if the text contains harmful/self-harm keywords or patterns.
    Returns True if harmful content is detected.
    """
    text_lower = text.lower()
    
    for keyword in HARM_KEYWORDS:
        if keyword in text_lower:
            return True
    
    for pattern in HARM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False

def check_emergency_content(text):
    """
    Check if the text contains emergency medical keywords.
    Returns True if emergency content is detected.
    """
    text_lower = text.lower()
    
    # Check for emergency keywords
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask():
    global model
    
    data = request.json
    question = data.get('question', '')
    
    if check_harmful_content(question):
        return jsonify({
            'answer': CRISIS_RESPONSE,
            'type': 'crisis'
        })
    
    # Check for emergency medical content
    if check_emergency_content(question):
        return jsonify({
            'answer': EMERGENCY_RESPONSE,
            'type': 'emergency'
        })
    
    if model is None:
        model = get_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    prompt_length = len(model.tokenizer.encode(prompt))
    
    with torch.no_grad():
        response = model.generate(prompt)
    
    tokens = model.tokenizer.encode(response)
    answer_tokens = tokens[prompt_length:]
    answer = model.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    if "\n\n" in answer:
        answer = answer.split("\n\n")[0].strip()
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return jsonify({
        'answer': answer,
        'type': 'normal'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
