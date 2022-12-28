import os
import io
import wave

import numpy as np

from flask import Flask, flash, render_template, request
from pydub import AudioSegment

import talk

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "user_data"

ALLOWED_EXTS = {'wav', 'mp3', 'ogg'}

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

    if request.method == 'POST':
        print("post request!")
        if request.form.get('reqtype') == 'upload':
            file = request.files['audio']
            if file.filename == '':
                print("No selected file!") 
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config["UPLOAD_DIR"], filename))
                print("successfully uploaded audio!")
            return 
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))







