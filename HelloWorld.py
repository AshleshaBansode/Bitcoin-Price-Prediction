# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:33:33 2018

@author: Ashlesha
"""

import Scraper_V_1
import mysql.connector
import PredictionFunctionFinal as pff
from flask import Flask, redirect
from flask import render_template
from flask import url_for, request
app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('/logins.html')    
    
@app.route('/success')
def success():    
        return render_template('try_popup.html')
    
@app.route('/daily',methods = ['POST', 'GET'])
def daily():
    date = request.form['date_from']
    pff.daily(date)
    return render_template('try_popup.html')

@app.route('/weekly',methods = ['POST', 'GET'])
def weekly():
    pff.weekly()
    return render_template('try_popup.html')

@app.route('/profile',methods = ['POST', 'GET'])
def profile():
    usr = request.form['username']
    pswrd = request.form['pwd']
    print("Username - " , usr)
    print("password - " , pswrd)
    cnx = mysql.connector.connect(user='ashlesha', password='welcome123',
                              host='127.0.0.1',
                              database='Bitcoin')
    cursor = cnx.cursor()
    query = "INSERT INTO users (name,password) values(%s,%s)"
    cursor.execute(query,(usr,pswrd,))
    cursor.execute('commit')
    return render_template('/logins_second.html', message = 'Added successfully')

@app.route('/<message>')
def error(message):
    return render_template('/logins_third.html', error = message)

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['name']
      pwd = request.form['password']
      cnx = mysql.connector.connect(user='ashlesha', password='welcome123',
                              host='127.0.0.1',
                              database='Bitcoin')
      cursor = cnx.cursor()
      query = "SELECT * FROM users WHERE name = (%s)"
      cursor.execute(query,(user,))
      row = cursor.fetchone()
      if row ==None:      
         return redirect(url_for('error', message = 'Invalid User'))
         cursor.close()
         cnx.close() 
      else:
          username,password= row[0], row[1]
          if password == pwd:
              return redirect(url_for('success'))
              cursor.close()
              cnx.close()
          else:
              return redirect(url_for('error', message = 'Wrong Password'))
              cursor.close()
              cnx.close()  
"""
    else:
      user = request.args.get('name')
      print("Username - " , user)
      return redirect(url_for('success',name = user))
"""

if __name__ == '__main__':    
    Scraper_V_1.getData()
    app.run(host = '0.0.0.0', port = '5000')
    