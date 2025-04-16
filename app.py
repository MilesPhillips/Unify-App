from flask import Flask
from flask import render_template

app = Flask(__name__)
@app.route('/home')
def home_page():
	return render_template('Home.html')

if __name__ == '__main__':
	app.run(debug=True)

#need a depth splash page to cusomize my web app, 
# to take you to the home page, (http://127.0.0.1:5000/home)