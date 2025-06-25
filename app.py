from flask import Flask
from flask import render_template
from flask import Flask, render_template, request, redirect, session, url_for, g
#from flask_bcrypt import Bcrypt
import sqlite3
import os
from werkzeug.utils import secure_filename

import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)


app = Flask(__name__)
#app.run(port=5000)


#in the static section add all of nate's java code so that it can be changed to typescript

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/profile')
def profile():
	return render_template('Profile.html')

@app.route('/login')
def login():
    return render_template('Login.html')

@app.route('/ai_coach')
def ai_coach():
    return render_template('AI_Coach.html')

@app.route('/contacts')
def contacts():
    return render_template('Contacts.html')

@app.route('/history')
def history():
    return render_template('History.html')

@app.route('/register')
def register():
    return render_template('Register.html')

@app.route('/index')
def indexPage():
    return render_template('Index.html')

@app.route('/splash')
def splashPage():
    return render_template('Splash.html')

@app.route('/record')
def record():
    return render_template('Record.html')


if __name__ == '__main__':
	app.run(debug=True)
 

#need a depth splash page to cusomize my web app, 
# to take you to the home page, (http://127.0.0.1:5000/home)

#login code:
app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)
DATABASE = 'database.db'


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


@app.route('/')
def home():
    return redirect(url_for('login'))


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


@app.route('/profile')
def profile():
    if 'user_id' in session:
        return render_template('profile.html', username=session['username'])
    return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    
    
#record code:
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRUSTED_USERS'] = {'user1': [], 'user2': []}  # simulate user inboxes

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    return redirect(url_for('record'))


@app.route('/record')
def record():
    return render_template('record.html', trusted_users=app.config['TRUSTED_USERS'].keys())


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


if __name__ == '__main__':
    app.run(debug=True)
    
#for new index.html code
@app.route('/')
def index():
    return render_template('index.html',
                           trusted_users=app.config['TRUSTED_USERS'].keys())
   
@app.route('/login', methods=['GET', 'POST'])
def login():
    # your existing login logic...
    if user and bcrypt.check_password_hash(user[2], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        return redirect(url_for('index'))
    
@app.route('/live-update', methods=['POST'])
def live_update():
    data = request.get_json()
    print("Received from DOM:", data.get("value"))
    print("data")
    return '', 204  # No content needed for async updates