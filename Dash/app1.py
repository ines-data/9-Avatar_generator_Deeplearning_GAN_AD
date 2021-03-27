import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



batch_size = 8
latent_dim = 128
height = 128
width = 128
channels = 3


model_path = 'generator.h5'
model = keras.models.load_model(model_path)

discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')


app.layout = html.Div(children=[
    html.H1(children='Generate Avatar'),

    html.Div(children='''
        Choose how many Avatar do you want to generate.
    '''),
    
    html.Button('Button 1', id='btn-nclicks-1', n_clicks=0),
    html.Div(id='container-button-timestamp')
])

@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'))

def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        image_random = np.random.normal(size=(batch_size,latent_dim))
        image_genereted = model.predict(image_random)
        for x in range(batch_size):
            ax = plt.subplot(4, 8, x+1)
            plt.imshow((image_genereted * 255).astype(np.uint8)[x])
            plt.axis("off")
            plt.savefig("assets/Avatar_images.png")
            break
    return html.Div([
        html.Img(src=app.get_asset_url('Avatar_images.png'), height=900, width=900, style={'padding-left':'25vw'})
        ])

    
if __name__ == '__main__':
    app.run_server(debug=True)
