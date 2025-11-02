SYSTEM_PROMPT = """You are a helpful and knowledgeable medical assistant. Provide accurate and concise answers to medical questions based on your training data. Always prioritize patient safety and evidence-based information. If you don't know the answer, say "I don't know."
""" 

USER_PROMPT = "Question: {question}\nAnswer:"

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


CRISIS_RESPONSE = """I'm concerned about what you're expressing. I can't help with this, but trained professionals can. Please reach out to:

• 988 Suicide & Crisis Lifeline: Call or text 988 (US)
• Crisis Text Line: Text HOME to 741741
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
• Emergency Services: Call 911 (US) or your local emergency number

You don't have to face this alone. People care about you and want to help."""

HARM_KEYWORDS = [
    'hang', 'hanging', 'hanged', 'noose', 'rope',
    'suicide', 'suicidal', 'kill myself', 'kill myself', 'kill me',
    'end my life', 'end my own life', 'end it',
    'overdose', 'overdosed', 'medication overdose',
    'cut myself', 'cutting', 'cut my wrists', 'slash', 'slit',
    'jump', 'jumping', 'jump off', 'jump from',
    'poison', 'poisoned', 'gas', 'carbon monoxide',
    'asphyxiate', 'asphyxiation', 'suffocate', 'drown',
    'stab', 'stabbing', 'blade', 'knife', 'stab myself',
    'harm myself', 'self harm', 'self-harm',
    
    'i want to die', 'i wanna die', 'i should die',
    'want to die', 'should die', 'better off dead',
    'better off gone', 'life is not worth',
    'no point living', 'not worth living', 'life is meaningless',
    'everyone would be better off', 'world be better without me',
    'nobody cares', 'no one would miss me', 'no reason to live',
    
    'how to end my life', 'how to kill myself',
    'ways to', 'methods to', 'best way to'
]

HARM_PATTERNS = [
    r'how\s+(do\s+i|to|can\s+i)\s+(hang|kill|cut|poison|overdose|slit)',
    r'how\s+(do\s+i|to|can\s+i)\s+end\s+my\s+life',
    r'how\s+(do\s+i|to|can\s+i)\s+harm\s+myself',
    r'best\s+way\s+to\s+(die|end\s+it)',
    r'how\s+to\s+(commit\s+)?suicide',
]

EMERGENCY_KEYWORDS = [
    'chest pain', 'chest tightness', 'heart attack',
    'difficulty breathing', 'can\'t breathe', 'shortness of breath',
    'severe bleeding', 'bleeding profusely',
    'loss of consciousness', 'unconscious', 'passed out',
    'severe allergic reaction', 'anaphylaxis',
    'poisoning', 'toxic', 'overdose',
    'severe burns', 'choking',
    'seizure', 'convulsing',
]

EMERGENCY_RESPONSE = """This may be a medical emergency. Call 911 or go to the nearest emergency room immediately."""
