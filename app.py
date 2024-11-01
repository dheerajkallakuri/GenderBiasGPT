from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from VietnameseBiasDetectionHelper import VietnameseEmbedder
from HindiBiasDetectionHelper import HindiEmbedder
import numpy as np
import json

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

def convert_numpy_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_float(item) for item in obj]
    return obj

def process_message(message, language):
    # Translate the input message to English
    translated = translate_message(message, language)

    # Generate sentence embedding based on the language
    if language == "Vietnamese":
        bias_score = vietnamese_embedder.get_gender_bias_score_of_sentence(message)
        print(bias_score)
        f_val,m_val,bias_tokens = round(abs(bias_score['female_bias_score']),2),round(abs(bias_score['male_bias_score']),2), bias_score['bias_tokens']
        converted_data = convert_numpy_to_float(bias_tokens)
    elif language == "Hindi":
        bias_score = hindi_embedder.get_gender_bias_score_of_sentence(message)
        f_val, m_val,bias_tokens = round(abs(bias_score['female_bias_score']),2),round(abs(bias_score['male_bias_score']),2), bias_score['bias_tokens']
        converted_data = convert_numpy_to_float(bias_tokens)
    else:
        raise ValueError("Unsupported language")

    print(converted_data)

    return {
        'translated': translated,
        'f_val': round(float(f_val),2),
        'm_val': round(float(m_val),2),
        'bias_tokens': json.dumps(converted_data)
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
    app.run()
