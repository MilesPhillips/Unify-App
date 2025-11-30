from flask import Flask, render_template, request, redirect, session, url_for, g, jsonify, send_from_directory
from flask_bcrypt import Bcrypt
import sqlite3
import os
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pdb
#for this add from "lib." to the import, add a lib folder to project and put all the stuff inside it except app.py
from lib.LLM import load_model_and_tokenizer, llm_generate_response, build_prompt, store_interaction
from dotenv import load_dotenv
from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForCausalLM

# A clear system instruction for the LLM
SYSTEM_INSTRUCTION = "You are a helpful assistant. Respond simply, clearly, and accurately."


# HW create and retrieve conversations had with the llm and save to db


load_dotenv() # Load variables from .env

database_url = os.getenv("https://github.com/MilesPhillips/Unify-App.git")


api_key = os.getenv("CLAUD_API_TOKEN")
#pdb .set_trace()

#Learn how to use copilet(vs code ai to the right) to suit you best!!!!!!!!!!

# langchain, llamaindex or haystack
# Uncomment and configure Firebase if needed
# import firebase_admin
# from firebase_admin import credentials
# cred = credentials.Certificate("path/to/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRUSTED_USERS'] = {'user1': [], 'user2': []}  # simulate user inboxes
DATABASE = 'database.db'

#Make sure this pipeline and messaging code works and reduce reduce reduce to make it simplified, get the llm to respond!!!

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
        db.commit()


# Routes
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/profile')
def profile():
    if 'user_id' in session:
        return render_template('profile.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and bcrypt.check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('profile'))
        return 'Invalid credentials!'
    return render_template('login.html')

@app.route('/ai_coach')
def ai_coach():
    return render_template('AI_Coach.html')

@app.route('/contacts')
def contacts():
    return render_template('Contacts.html')

@app.route('/history')
def history():
    return render_template('History.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        try:
            db = get_db()
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists!'
    return render_template('register.html')

@app.route('/index')
def indexPage():
    return render_template('index.html', trusted_users=app.config['TRUSTED_USERS'].keys())

@app.route('/splash')
def splashPage():
    return render_template('Splash.html')

@app.route('/record')
def record():
    return render_template('record.html', trusted_users=app.config['TRUSTED_USERS'].keys())

@app.route('/index_transcripter')
def index_transcripter():
    return render_template('index_transcripter.html')

@app.route('/connect')
def connect():
    return render_template('connect.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file provided', 400

    video = request.files['video']
    trusted_user = request.form['trusted_user']

    if video and trusted_user in app.config['TRUSTED_USERS']:
        filename = secure_filename(video.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(save_path)
        app.config['TRUSTED_USERS'][trusted_user].append(filename)
        return 'Video uploaded successfully'
    return 'Invalid upload', 400

@app.route('/inbox/<username>')
def inbox(username):
    if username not in app.config['TRUSTED_USERS']:
        return 'User not found', 404
    files = app.config['TRUSTED_USERS'][username]
    return render_template('inbox.html', files=files, username=username)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/live-update', methods=['POST'])
def live_update():
    data = request.get_json()
    print("Received from DOM:", data.get("value"))
    return '', 204  # No content needed for async updates

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
   
#New transcript parsing code:
# Load model and tokenizer once
#models
#model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"#"meta-llama/Meta-Llama-3-8B"


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
model.eval()


@app.route("/transcribe", methods=["POST"])
def api_transcribe():
    data = request.get_json(silent=True)
    if not data or "transcript" not in data:
        return jsonify({"status": "error", "message": "Missing 'transcript'"}), 400
    transcript = data["transcript"].strip()
    pdb.set_trace()
    
    print(transcript)
  
    return jsonify({"status": "ok", "message": "transcript received", "transcript": transcript})


#New test code
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("transcript") or data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    # initialize per-user session history
    if "history" not in session:
        session["history"] = []

    # build prompt with history
    prompt = build_prompt(session["history"], user_msg, SYSTEM_INSTRUCTION)
    if prompt == build_prompt(session["history"], user_msg, SYSTEM_INSTRUCTION):
        print("Prompt matches build_prompt output.")
        

    # generate with local model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract just the assistant continuation after the final "Assistant:"
    reply = full_text.split("Assistant:")[-1].strip()

    print("LLM reply:", reply)  # <-- Add this line


    # update memory + persist
    session["history"].append({"role": "user", "content": user_msg})
    session["history"].append({"role": "assistant", "content": reply})
    session.modified = True

    store_interaction(user_msg, reply)

    return jsonify({ "response": reply })



# build prompt function reused from LLM.py(may need to adjust import paths if moved to lib/  , and be updated to match new version there)
@app.route("/index_transcripter", methods=["POST"])
def transcribe_legacy():
    data = request.get_json(silent=True) or {}
    transcript = data.get("transcript", "").strip()
    model, tokenizer = load_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if not transcript:
        return jsonify({"error": "No transcript"}), 400

    # Reuse the chat logic
    if "history" not in session:
        session["history"] = []

    prompt = build_prompt(session["history"], transcript)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    session["history"].append({"role": "user", "content": transcript})
    session["history"].append({"role": "assistant", "content": reply})
    session.modified = True
    store_interaction(transcript, reply)

    return jsonify({"response": reply})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    