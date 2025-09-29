import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr
import re
import os  
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the file is in the same directory.")
    exit()


stop_words = set(['a','an','the','and','or','is','are','of','to','in','for','on','with','you','we','our','your'])

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [t.lower() for t in text.split() if t.lower() not in stop_words]
    return " ".join(tokens)

df['processed'] = df['Question'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed'])

def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, X)
    return df['Answer'].iloc[similarity.argmax()]

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for SpaCy...")
    
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

engine = pyttsx3.init()

recognizer = sr.Recognizer()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response_text = ""
    entities = []

    if user_message:
        response_text = get_response(user_message)
        entities = extract_entities(user_message)

    return jsonify({'response': response_text, 'entities': entities})

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    response_data = {'response': "Sorry, I didn't catch that.", 'entities': []}
    
    temp_audio_path = "temp_uploaded_audio.wav" 
    
    try:
        audio_file = request.files['audio']
        
        audio_file.save(temp_audio_path)

        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)

        user_input = recognizer.recognize_google(audio)
        print(f"Voice Input: {user_input}")

        if user_input.lower() == "quit":
            response_data['response'] = "Exiting chatbot..."
        else:
            response_text = get_response(user_input)
            entities = extract_entities(user_input)
            response_data['response'] = response_text
            response_data['entities'] = entities

    except sr.UnknownValueError:
        pass  
    except sr.RequestError as e:
        response_data['response'] = f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        response_data['response'] = f"An error occurred: {e}"
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return jsonify(response_data)

@app.route('/speak', methods=['POST'])
def speak():
    text_to_speak = request.json.get('text')
    if text_to_speak:
        engine.say(text_to_speak)
        engine.runAndWait()
    return jsonify({'status': 'spoken'})

if __name__ == '__main__':
    app.run(debug=True)