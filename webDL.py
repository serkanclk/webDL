from flask import Flask, render_template, request
from flask.helpers import flash, url_for
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from werkzeug.utils import redirect

app = Flask(__name__)
model = InceptionResNetV2()

@app.route('/',methods=['GET'])

def initiate():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    if not imagefile.filename =='':
        image_path = "./static/images/" + imagefile.filename
        imagefile.save(image_path)
        image = load_img(image_path, target_size=(299,299))
        image = img_to_array(image)
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image = preprocess_input(image)
        predict = model.predict(image)
        label = decode_predictions(predict)
        label = label[0][0]
        # [1] -> class_name | [2] -> top-1_acc_score
        classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    
        return render_template('index.html',prediction = classification, filename=imagefile.filename)
    else:
        return render_template('index.html')

@app.route('/images/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='/images/'+filename),code=301)

if __name__ == '__main__':
    app.run(port=3000,debug=True)
# 127.0.0.1:3000 or localhost:3000