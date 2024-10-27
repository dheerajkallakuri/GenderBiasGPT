from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from VietnameseBiasDetectionHelper import VietnameseEmbedder
from HindiBiasDetectionHelper import HindiEmbedder
import os

PORT = int(os.getenv("PORT")) 

vietnamese_embedder = VietnameseEmbedder()
hindi_embedder = HindiEmbedder()

app = Flask(__name__)

@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.json  # Get JSON data from the request
    message = data.get('message')
    language = data.get('language')
    
    if not message or not language:
        return jsonify({'error': 'Message and language are required.'}), 400

    # Process the input message based on the language
    
    response = process_message(message, language)
    print(response)
    if response: 
        return jsonify({'response': response}), 200
    else:
        return "Error"

def process_message(message, language):
    # Translate the input message to English
    translated = translate_message(message, language)

    # Generate sentence embedding based on the language
    if language == "Vietnamese":
        bias_score = vietnamese_embedder.get_gender_bias_score_of_sentence(message)
        print(bias_score)
        f_val,m_val,bias_tokens = round(abs(bias_score['female_bias_score']),2),round(abs(bias_score['male_bias_score']),2), bias_score['bias_tokens']
    elif language == "Hindi":
        bias_score = hindi_embedder.get_gender_bias_score_of_sentence(message)
        f_val, m_val,bias_tokens = round(abs(bias_score['female_bias_score']),2),round(abs(bias_score['male_bias_score']),2), bias_score['bias_tokens']
    else:
        raise ValueError("Unsupported language")

    print(f_val)
    print(m_val)
    print(bias_tokens)

    return {
        'translated': translated,
        'f_val': round(float(f_val),2),
        'm_val': round(float(m_val),2)
    }

def translate_message(message, language):
    if language == "Vietnamese":
        translated_message = GoogleTranslator(source='vi', target='en').translate(message)
        return translated_message
    elif language == "Hindi":
        translated_message = GoogleTranslator(source='hi', target='en').translate(message)
        return translated_message
    else:
        raise ValueError(f"Unsupported language: {language}")

if __name__ == '__main__':
    app.run(port=PORT)
