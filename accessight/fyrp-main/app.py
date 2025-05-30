from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
import sqlite3
import os
import platform

import mediapipe as mp

import numpy as np
import pickle
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
import pandas as pd
import pyttsx3
import tempfile
import speech_recognition as sr
from gtts import gTTS
from tensorflow.keras.models import load_model
import threading
import uuid
import cv2
from transformers import T5Tokenizer, T5ForConditionalGeneration



#from transformers import pipeline
#import uuid
T5_MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
model_sum = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)


app = Flask(__name__, static_folder='static')
app.secret_key = '123'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
audio_folder = os.path.join(app.static_folder, 'audio')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

#summarizer = pipeline('summarization')
model = load_model('chatbot_model.h5')

with open('intents.json') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)


wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def init_db():
    with sqlite3.connect("data.db") as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )''')
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

##def summarize_text(full_text):
##    if full_text.strip():
##        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
##        summary = ""
##        for chunk in chunks:
##            try:
##                result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
##                summary += result[0]['summary_text'] + " "
##            except Exception:
##                summary += "[Summarization failed for chunk] "
##        return summary.strip()
##    return "No clear text found to summarize."


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model_sum = T5ForConditionalGeneration.from_pretrained("t5-small")

from nltk.tokenize import sent_tokenize

def remove_duplicate_sentences(text):
    seen = set()
    result = []
    for sentence in sent_tokenize(text):
        if sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    return " ".join(result)

def summarize_text(full_text):
    inputs = tokenizer("summarize: " + full_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model_sum.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    texts =  tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    cleaned_text = remove_duplicate_sentences(texts)
    return cleaned_text


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [wnl.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(msg):
    print(f"Message received: {msg}")
    bow = bag_of_words(msg, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"Response found: {response}")
                return response
    print("No response found")
    return "Sorry, I didn't understand that."

@app.route('/')
def login():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    try:
        with sqlite3.connect("data.db") as conn:
            conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
        return redirect(url_for('login'))
    except sqlite3.IntegrityError:
        return "Email already exists!"

@app.route('/login', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']
    with sqlite3.connect("data.db") as conn:
        cursor = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
    if user:
        session['user_id'] = user[0]
        return redirect(url_for('bot'))
    else:
        return "Invalid login credentials"

@app.route('/bot')
def bot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('bot.html')

@app.route('/pdf_text', methods=['POST'])
def summarize_pdf_text():
    pdf_file = request.files['pdf']
    full_text = ""
    if pdf_file and allowed_file(pdf_file.filename):
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
        summary = summarize_text(full_text)
        return jsonify({"original": full_text, "summary": summary})
    else:
        return "Invalid or missing PDF file."

from googletrans import Translator  

translator = Translator()  

@app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.form['message']
    selected_lang = request.form.get('language', 'en')  

   
    response = get_response(message)

    
    try:
        translated_text = translator.translate(response, dest=selected_lang).text
    except Exception as e:
        print(f"Translation error: {e}")
        translated_text = response  

   
    audio_filename = f"response_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(audio_folder, audio_filename)

    try:
        tts = gTTS(text=translated_text, lang=selected_lang)
        tts.save(audio_path)
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({
            "response": translated_text,
            "audio_file": "",
            "error": f"TTS failed: {str(e)}"
        })

    
    return jsonify({
        "response": translated_text,
        "audio_file": f"/static/audio/{audio_filename}"
    })


@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    text = request.form['text']                                                
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_path = temp_file.name
    temp_file.close()
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    return send_file(temp_path, mimetype="audio/mpeg", as_attachment=False)

r = sr.Recognizer()
mic = sr.Microphone()
lock = threading.Lock()

def speak1():
    print("Speak Now...")
    with lock:
        try:
            with mic as audio_file:
                r.adjust_for_ambient_noise(audio_file)
                audio = r.listen(audio_file)
                print("Audio captured...")
                text = r.recognize_google(audio)
                print("Text from speech:", text)
                return text.lower()
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None

@app.route('/speak', methods=['GET', 'POST'])
def speak():
    speech = speak1()
    if not speech:
        return render_template('bot.html', speech="Error: Could not understand the audio", result="")
    result = get_response(speech)
    tts = gTTS(result)
    audio_file_path = os.path.join(audio_folder, 'response.mp3')
    tts.save(audio_file_path)
    audio_url = '/static/audio/response.mp3'
    return jsonify({"speech": speech, "result": result, "audio_file": audio_url})

with open('model_ASL.pkl', 'rb') as f:
    model_sign = pickle.load(f)
    print('work')


mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global detected_letters, last_detected_letter, whole_word_text, body_language_class
    global current_keyword

    
    detected_letters = [] 
    whole_word_text = ""   
    current_keyword = ""   
    last_detected_letter = None  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return render_template("index.html")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            body_language_class = "None"
            current_keyword = None

            if results.left_hand_landmarks:
                pose = results.left_hand_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                row = pose_row
                X = pd.DataFrame([row])
                body_language_class = model_sign.predict(X)[0]
                print(body_language_class)
                current_detected_letter = body_language_class

    
            key = cv2.waitKey(5)

            if key == 105: 
                if body_language_class != "None":
                    detected_letters.append(body_language_class) 
                    last_detected_letter = body_language_class.split(' ')[0] 
                    whole_word_text = "".join(detected_letters)  

            elif key == ord('s'):  
                detected_letters.append(" ") 
                whole_word_text = "".join(detected_letters) 

            elif key == ord('d'):  
                detected_letters.clear() 
                whole_word_text = ""  
                last_detected_letter = None 

            elif key == 13: 
                user_message = whole_word_text.strip()
                print(f"User message sent: {user_message}")
                bot_reply = get_response(user_message)
                print(f"Bot reply: {bot_reply}")
                cap.release()
                cv2.destroyAllWindows()
                return render_template("bot.html", user_message=user_message, bot_reply=bot_reply)


            elif cv2.waitKey(10) & 0xFF == ord('q'):  
                break

        
            last_letter_text = f'Letter: {last_detected_letter}' if last_detected_letter else 'Letter: None'

            cv2.putText(image, whole_word_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, last_letter_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Detect Sign", image)

        cap.release()
        cv2.destroyAllWindows()

    return render_template("bot.html")




from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
wnl = WordNetLemmatizer()
nltk.download('wordnet')




def text_to_sign_videos(user_text):
        processed = []  
        tokens_sign_lan = []  
        stop = nltk.corpus.stopwords.words('english')
        stop_words = ['@','#',"http",":","is","the","are","am","a","it","was","were","an",",",".","?","!",";","/"]
        for i in stop_words:
            stop.append(i)
        
        tokenized_text = nltk.tokenize.word_tokenize(user_text)
        lemmed = [wnl.lemmatize(word) for word in tokenized_text]
        
        for i in lemmed:
            if i == "i" or i == "I":
                processed.append("me")
            #elif i not in stop:
            else:
                i = i.lower()
                processed.append(i)
        
        assets_list = ['0.mp4', '1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4', '6.mp4', '7.mp4', '8.mp4', '9.mp4', 'a.mp4', 
                       'after.mp4', 'again.mp4', 'against.mp4', 'age.mp4', 'all.mp4', 'alone.mp4', 'also.mp4', 'and.mp4', 'ask.mp4',
                       'at.mp4', 'b.mp4', 'be.mp4', 'beautiful.mp4', 'before.mp4', 'best.mp4', 'better.mp4', 'busy.mp4', 'but.mp4', 
                       'bye.mp4', 'c.mp4', 'can.mp4', 'cannot.mp4', 'change.mp4', 'college.mp4', 'come.mp4', 'computer.mp4', 'd.mp4', 
                       'day.mp4', 'distance.mp4', 'do not.mp4', 'do.mp4', 'does not.mp4', 'e.mp4', 'eat.mp4', 'engineer.mp4', 'f.mp4', 
                       'fight.mp4', 'finish.mp4', 'from.mp4', 'g.mp4', 'glitter.mp4', 'go.mp4', 'god.mp4', 'gold.mp4', 'good.mp4', 
                       'great.mp4', 'h.mp4', 'hand.mp4', 'hands.mp4', 'happy.mp4', 'hello.mp4', 'help.mp4', 'her.mp4', 'here.mp4', 
                       'his.mp4', 'home.mp4', 'homepage.mp4', 'how.mp4', 'i.mp4', 'invent.mp4', 'it.mp4', 'j.mp4', 'k.mp4', 'keep.mp4', 
                       'l.mp4', 'language.mp4', 'laugh.mp4', 'learn.mp4', 'm.mp4', 'me.mp4', 'mic3.png', 'more.mp4', 'my.mp4', 'n.mp4', 
                       'name.mp4', 'next.mp4', 'not.mp4', 'now.mp4', 'o.mp4', 'of.mp4', 'on.mp4', 'our.mp4', 'out.mp4', 'p.mp4', 
                       'pretty.mp4', 'q.mp4', 'r.mp4', 'right.mp4', 's.mp4', 'sad.mp4', 'safe.mp4', 'see.mp4', 'self.mp4', 'sign.mp4', 
                       'sing.mp4', 'so.mp4', 'sound.mp4', 'stay.mp4', 'study.mp4', 't.mp4', 'talk.mp4', 'television.mp4', 'thank you.mp4', 
                       'thank.mp4', 'that.mp4', 'they.mp4', 'this.mp4', 'those.mp4', 'time.mp4', 'to.mp4', 'type.mp4', 'u.mp4', 'us.mp4', 
                       'v.mp4', 'w.mp4', 'walk.mp4', 'wash.mp4', 'way.mp4', 'we.mp4', 'welcome.mp4', 'what.mp4', 'when.mp4', 'where.mp4', 
                       'which.mp4', 'who.mp4', 'whole.mp4', 'whose.mp4', 'why.mp4', 'will.mp4', 'with.mp4', 'without.mp4', 'words.mp4', 
                       'work.mp4', 'world.mp4', 'wrong.mp4', 'x.mp4', 'y.mp4', 'you.mp4', 'your.mp4', 'yourself.mp4', 'z.mp4']
        
        for word in processed:
            string = str(word + ".mp4")
            if string in assets_list:
                tokens_sign_lan.append("assets/" + string)
            else:
                for j in word:
                    tokens_sign_lan.append("assets/" + j + ".mp4")
        return tokens_sign_lan  

@app.route('/learning_path', methods=['GET'])
def learning_path():
    return render_template('bot.html')

@app.route('/learning_path', methods=['POST'])
def learning_path_post():
    input_text = request.form.get('input_text', '')
    video_urls = []
    videos = text_to_sign_videos(input_text)
    video_urls = [url_for('static', filename=f'/{video}') for video in videos]
    return jsonify({'video_urls': video_urls})

if __name__ == '__main__':
    app.run(debug=False, port=1230)


