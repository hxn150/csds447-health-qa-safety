from flask import Flask, request, jsonify, render_template_string
import torch
import re

app = Flask(__name__)

model = None
safety_gate = None

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Q&A Interface</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: linear-gradient(to left top, #f7edf4, #f8f2f8, #f9f6fb, #fcfbfd, #ffffff);
        }
        .header {
            position: absolute;
            top: 20px;
            left: 20px;
            color: black;
            font-size: 24px;
            font-weight: 400;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 80px 20px 120px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message-bubble {
            max-width: 70%;
            padding: 16px 20px;
            border-radius: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            line-height: 1.6;
        }
        .user-message {
            background: #e3f2fd;
            align-self: flex-end;
        }
        .bot-message {
            background: white;
            align-self: flex-start;
        }
        .bot-message p {
            margin-bottom: 12px;
        }
        .bot-message p:last-child {
            margin-bottom: 0;
        }
        .bot-message ul, .bot-message ol {
            margin: 8px 0;
            padding-left: 24px;
        }
        .bot-message li {
            margin: 6px 0;
        }
        .bot-message strong {
            font-weight: 600;
            color: #333;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }
        .input-wrapper {
            width: 100%;
            max-width: 600px;
            display: flex;
            gap: 10px;
            background: white;
            border-radius: 30px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        input {
            flex: 1;
            border: none;
            outline: none;
            padding: 12px 16px;
            font-size: 15px;
        }
        button {
            background: #F7EDF4;
            color: black;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        button:hover { background: #e0d5dd; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .placeholder-text {
            color: black;
            text-align: center;
            margin-top: 120px;
            font-size: 24px;
            font-weight: 300;
        }
    </style>
</head>
<body>
    <div class="header">Dr.Darla</div>
    <div class="chat-container" id="chatContainer">
        <div class="placeholder-text">Ask Darla anything</div>
    </div>
    <div class="input-container">
        <div class="input-wrapper">
            <input type="text" id="userInput" placeholder="Ask me anything about medical questions">
            <button id="sendBtn" onclick="sendMessage()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
            </button>
        </div>
    </div>
    <script>
        function formatResponse(text) {
            let formatted = text;
            
            // Split into paragraphs
            let paragraphs = formatted.split(/\\n\\n+/);
            
            let html = '';
            paragraphs.forEach(para => {
                para = para.trim();
                if (!para) return;
                
                // Check if it's a list (lines starting with -, *, or numbers)
                const lines = para.split('\\n');
                const isList = lines.length > 1 && lines.every(line => 
                    /^[-*•]\\s/.test(line.trim()) || /^\\d+\\.\\s/.test(line.trim())
                );
                
                if (isList) {
                    const isOrdered = /^\\d+\\.\\s/.test(lines[0].trim());
                    const tag = isOrdered ? 'ol' : 'ul';
                    html += `<${tag}>`;
                    lines.forEach(line => {
                        line = line.trim().replace(/^[-*•]\\s/, '').replace(/^\\d+\\.\\s/, '');
                        if (line) {
                            html += `<li>${line}</li>`;
                        }
                    });
                    html += `</${tag}>`;
                } else {
                    // Regular paragraph
                    html += `<p>${para}</p>`;
                }
            });
            
            return html;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            const message = input.value.trim();
            if (!message) return;
            
            const container = document.getElementById('chatContainer');
            const placeholder = container.querySelector('.placeholder-text');
            if (placeholder) placeholder.remove();
            
            const userBubble = document.createElement('div');
            userBubble.className = 'message-bubble user-message';
            userBubble.textContent = message;
            container.appendChild(userBubble);
            
            input.value = '';
            sendBtn.disabled = true;
            container.scrollTop = container.scrollHeight;
            
            const botBubble = document.createElement('div');
            botBubble.className = 'message-bubble bot-message';
            botBubble.textContent = 'Thinking...';
            container.appendChild(botBubble);
            container.scrollTop = container.scrollHeight;
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: message})
                });
                const data = await response.json();
                botBubble.innerHTML = formatResponse(data.answer);
            } catch (error) {
                botBubble.textContent = 'Error: Could not get response';
            }
            
            sendBtn.disabled = false;
            container.scrollTop = container.scrollHeight;
        }
        
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

def check_harmful_content(text):
    try:
        from config import HARM_KEYWORDS, HARM_PATTERNS
        text_lower = text.lower()
        for keyword in HARM_KEYWORDS:
            if keyword in text_lower:
                return True
        for pattern in HARM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
    except Exception as e:
        print(f"Error in check_harmful_content: {e}")
    return False

def check_emergency_content(text):
    try:
        from config import EMERGENCY_KEYWORDS
        text_lower = text.lower()
        for keyword in EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                return True
    except Exception as e:
        print(f"Error in check_emergency_content: {e}")
    return False

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask():
    global model, safety_gate
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if safety_gate is None:
            try:
                from safety_gate import SafetyGate
                safety_gate = SafetyGate()
                print("Safety gate loaded successfully")
            except Exception as e:
                print(f"Could not load safety gate: {e}")
        
        if safety_gate is not None:
            try:
                safety_result = safety_gate.check_safety(question)
                print(f"Question: {question[:50]}... -> {safety_result['label']} ({safety_result['confidence']:.2f})")
                
                if not safety_result['allowed']:
                    from config import CRISIS_RESPONSE
                    return jsonify({'answer': CRISIS_RESPONSE, 'type': 'crisis'})
            except Exception as e:
                print(f"Safety check error: {e}")
        
        if check_harmful_content(question):
            from config import CRISIS_RESPONSE
            return jsonify({'answer': CRISIS_RESPONSE, 'type': 'crisis'})
        
        if check_emergency_content(question):
            from config import EMERGENCY_RESPONSE
            return jsonify({'answer': EMERGENCY_RESPONSE, 'type': 'emergency'})
        
        if model is None:
            try:
                from registry import get_model
                from config import MODEL
                print("Loading model...")
                model = get_model(MODEL)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Could not load model: {e}")
                return jsonify({'answer': 'Error: Model could not be loaded', 'type': 'error'})
        
        from config import SYSTEM_PROMPT
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
        
        with torch.no_grad():
            response = model.generate(prompt, max_new_tokens=512)
        
        if "Answer:" in response:
            answer = response.split("Answer:", 1)[1].strip()
        else:
            answer = response.strip()
        
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({'answer': answer, 'type': 'normal'})
    
    except Exception as e:
        print(f"Error in /ask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'answer': f'Error: {str(e)}', 'type': 'error'})

if __name__ == '__main__':
    print("Starting Flask app on http://localhost:5000")
    app.run(debug=True, port=5000)