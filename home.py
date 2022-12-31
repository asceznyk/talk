import os
import io
import wave
import json

import tempfile
import numpy as np

from flask import Flask, flash, render_template, request
from pydub import AudioSegment

import talk

app = Flask(__name__)

tempdir = "/tmp/"
for fname in os.listdir(tempdir): 
    if fname.startswith("talk_user_data_"): 
        app.config["UPLOAD_DIR"] = f"{tempdir}{fname}"
        break
else: 
    app.config["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="talk_user_data_")

print(app.config["UPLOAD_DIR"])

allowed_exts = {'wav', 'mp3', 'ogg'}

model = talk.load_model("assets/tiny.pt")

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

    if request.method == 'POST':
        file = request.files['audio']
        if file.filename == '':
            print("No selected file!") 

        if file and allowed_file(file.filename): 
            to_annotate = os.path.join(app.config["UPLOAD_DIR"], file.filename)
            file.save(to_annotate)
            options = DecodingOptions(task=request.form.get('task'))
            mel = talk.pad_or_trim(talk.log_mel_spec(to_annotate), length=2*model.dims.n_audio_ctx) 
            result = model.decode(mel, options)
            text = result.text
        else:
            text = f"incorrect file format, allowed exts {str(allowed_exts)[1:-1]}"

            return json.dumps({"text":text})
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))


