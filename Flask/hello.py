"""
 Flask HelloWorld
"""
from flask import Flask
app = Flask(__name__)

# home 
@app.route('/')
def hello():
  return 'Hello World'

# home/tokyo
@app.route('/tokyo')
def tokyo():
  return 'Hello Tokyo'


if __name__ == '__main__':
  app.run(debug=True)
  
