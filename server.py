import os
import cv2
import glob
from flask import Flask, render_template, request, redirect, url_for
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
import pandas as pd
import wgetter
from keras.models import load_model
import joblib
from flask import jsonify
from sklearn.metrics import accuracy_score
import csv

UPLOAD_FOLDER = 'user_uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def check_or_make_folder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)


check_or_make_folder(UPLOAD_FOLDER)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('saved_model/dermosis.h5')
mapper = joblib.load('saved_model/dermosis_mapping.pkl')
mapper = {v: k for k, v in mapper.items()}

EXTRA_DETAILS_LOCATION = "disease_extra_details_2.csv"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def main():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        pic = preprocess_single_image(filepath)
        pred_class = model.predict_classes(pic)[0]

        print("d:  " + str(pred_class))
        pred_class_name = get_pred_class_name(pred_class)
        pred_class_extra_details_dic = get_pred_class_extra_details(pred_class_name)
        print("Predicted class is {}".format(pred_class_name))

        if request.method == 'POST':
            return jsonify(
        diseaseName=pred_class_name,
        precautions=pred_class_extra_details_dic.strip("\n")
    )

        return pred_class_extra_details_dic

    return "upload rejected"



@app.route("/display")
def test():
    dic = joblib.load("diseaseinfo.pkl")
    return dic




def get_pred_class_extra_details(pred_class_name):
    """df = load_and_format_extra_details_csv()
    df = df[df["Disease"] == pred_class_name]
    print(df+" ")"""
    data=""

    csv_file = csv.reader(open(EXTRA_DETAILS_LOCATION, "r"), delimiter=",")


    for row in csv_file:

        if pred_class_name== row[1]:
            row = row[2].strip('. ')

            return  row


def get_pred_class_name(pred_class_number):
    global mapper
    return mapper[pred_class_number]


def load_and_format_extra_details_csv():
    global EXTRA_DETAILS_LOCATION
    df = pd.read_csv(EXTRA_DETAILS_LOCATION)
    df["Disease"] = [x.replace(' ', '%20') for x in df["Disease"]]
    return df


def preprocess_single_image(filepath):
    pic = cv2.imread(filepath)
    pic = cv2.resize(pic, (100, 100))
    pic = pic.astype('float32')
    pic /= 255
    pic = pic.reshape(-1, 100, 100, 3)
    return pic


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
