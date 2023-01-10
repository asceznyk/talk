import os
import io
import wave
import json

import tempfile
import traceback
import numpy as np

from flask import Flask, flash, render_template, request
from pydub import AudioSegment
from scipy.io import wavfile

from talk import load_model, log_mel_spec, pad_or_trim, DecodingOptions

app = Flask(__name__)

tempdir = "/tmp/"
for fname in os.listdir(tempdir): 
    if fname.startswith("talk_user_data_"): 
        app.config["UPLOAD_DIR"] = f"{tempdir}{fname}"
        break
else: 
    app.config["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="talk_user_data_")

allowed_exts = {'wav', 'mp3', 'ogg', 'webm'}
base_path = "assets/tiny.pt"
model, _ = load_model(base_path)

@app.route("/checkpoint/", methods=['POST'])
def get_model():
    global model
    del model
    try:
        model, status = load_model(f"assets/{request.form.get('checkpoint')}.pt")
    except:
        model, _ = load_model(base_path)
        status = "model loading has failed, loaded tiny model instead"

    return json.dumps({"status":status})

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

    if request.method == 'POST':
        language = "en"
        file = request.files['audio']
        if file and allowed_file(file.filename): 
            to_annotate = os.path.join(app.config["UPLOAD_DIR"], file.filename)
            file.save(to_annotate)

            inp = request.form['prompt']
            options = DecodingOptions(
                task = request.form['task'], 
                language = request.form['language'],
                log_tensors = True
                prompt = None if inp == '' else inp 
            )
            print(f"input options task={options.task}, language={options.language}")
            print(f"log_tensors={options.log_tensors}")

            webm = AudioSegment.from_file(to_annotate, 'webm')
            to_annotate_wav = to_annotate.replace('webm', 'wav')
            webm.export(to_annotate_wav, format='wav') 
            
            mel = log_mel_spec(to_annotate_wav) 
            mel = pad_or_trim(mel, length=2*model.dims.n_audio_ctx) 

            print(f"input audio shape: {mel.shape}")
            result = model.decode(mel, options)
            text, language = result.text, result.language
            print(options.prompt)
            print(text, language)
        else:
            text = f"incorrect file format, allowed exts {str(allowed_exts)[1:-1]}"

        return json.dumps({"text":text, "language":language})
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', ssl_context='adhoc', port=os.environ.get('PORT', 5000))


