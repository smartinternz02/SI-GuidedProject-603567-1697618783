# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:25:35 2023

@author: ajaypoonia
"""
#import the libraries
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# load model
app = Flask(__name__, static_url_path='/static')

model = pickle.load(open(r'temperature.pkl', 'rb'))
app = Flask(__name__)

# Render HTML pages
@app.route("/")
def home():
    return render_template("index.html")

#contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Change the route to render predict.html
@app.route("/predict")
def predict():
    return render_template("predict.html")

# Retrieve value from UserInterface
@app.route('/output', methods=['post', 'get'])
def output():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    input_feature = [np.array(input_feature)]
    print(input_feature)
    names = ['CO2_(dinning-room)', 'CO2_room', 'Relative_humidity_(dinning-room)', 'Relative_humidity_room',
             'Lighting_(dinning-room)', 'Lighting_room', 'Meteo_Rain', 'Meteo_Sun_dusk', 'Meteo_Wind',
             'Meteo_Sun_light_in_west_facade', 'Meteo_Sun_light_in_east_facade', 'Meteo_Sun_light_in_south_facade',
             'Meteo_Sun_irradiance', 'Outdoor_relative_humidity_Sensor']
    print(names)
    data = pd.DataFrame(input_feature, columns=names)
    print(data)
    prediction = model.predict(data)
    print(prediction)
    return render_template('predict.html', result="Your room temperature will be:  " + str(np.round(prediction[0])))

# Main Function
if __name__ == '__main__':
    app.run(debug=True)
