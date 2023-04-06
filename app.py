from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5', compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read the image file from the request object
    image_file = request.files['file']
    
    # Open the image file using PIL
    image = Image.open(image_file)
    
    # Preprocess the image
    image = image.resize((625, 100)) # Resize the image to the input size of your model
    #image = np.array(image) / 255.0 # Normalize the pixel values
    image = np.expand_dims(image, axis=0) # Add batch dimension
    
    # Make a prediction using your model
    predictions = model.predict(image)
    
    if predictions[0,0]>0.6:
        result="This is type1 Crazing defect"
    elif predictions[0,1]>0.6:
        result="This is type2 Scratching defect"
    elif predictions[0,2]>0.6:
        result="This is type3 Inclusion defect"
    elif predictions[0,3]>0.6:
        result="This is type4 Hole defect"
    else:
        result="The steel is not defected"
    
    # Return the prediction as a JSON object
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)