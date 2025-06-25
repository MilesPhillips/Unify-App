from flask import Flask, render_template, request, redirect, session, url_for, g, jsonify, send_from_directory
from flask_bcrypt import Bcrypt
import sqlite3
import os
from werkzeug.utils import secure_filename

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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    transcript = data.get('transcript')
    print(f"Received transcript: {transcript}")
    
    # Here you can process the transcript as needed
    # For example, save to database, analyze, etc.
    
    return jsonify({'status': 'success', 'message': 'Transcript received', 'transcript': transcript})

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


if __name__ == '__main__':
    init_db()
    app.run(debug=True)