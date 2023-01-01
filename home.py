import os
import io
import wave
import json

import tempfile
import numpy as np

from flask import Flask, flash, render_template, request
from pydub import AudioSegment

from talk import load_model, log_mel_spec, pad_or_trim, DecodingOptions

app = Flask(__name__)

tempdir = "/tmp/"
for fname in os.listdir(tempdir): 
    if fname.startswith("talk_user_data_"): 
        app.config["UPLOAD_DIR"] = f"{tempdir}{fname}"
        break
else: 
    app.config["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="talk_user_data_")

allowed_exts = {'wav', 'mp3', 'ogg'}
model = load_model("assets/tiny.pt")

@app.route("/checkpoint/", methods=['POST'])
def get_model(): 
    model, status = load_model(f"assets/{request.form.get('checkpoint')}.pt")
    return json.dumps({"status":status})

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

    if request.method == 'POST':
        language = "en"
        file = request.files['audio']
        if file.filename == '':
            print("No selected file!") 
        if file and allowed_file(file.filename): 
            to_annotate = os.path.join(app.config["UPLOAD_DIR"], file.filename)
            file.save(to_annotate)
            options = DecodingOptions(task=request.form.get('task'))
            mel = pad_or_trim(log_mel_spec(to_annotate), length=2*model.dims.n_audio_ctx) 
            result = model.decode(mel, options)
            text, language = result.text, result.language
            print(text, language)
        else:
            text = f"incorrect file format, allowed exts {str(allowed_exts)[1:-1]}"

        return json.dumps({"text":text, "language":language})
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))


