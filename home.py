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

from talk import load_model, log_mel_spec, pad_or_trim, decode, DecodingOptions

app = Flask(__name__)

tempdir = "/tmp/"
for fname in os.listdir(tempdir): 
    if fname.startswith("talk_user_data_"): 
        app.config["UPLOAD_DIR"] = f"{tempdir}{fname}"
        break
else: 
    app.config["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="talk_user_data_")

allowed_exts = {'wav', 'mp3', 'ogg', 'webm'}
base_path = "assets/base.pt"

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

    try:
        if request.method == 'POST':
            model, _ = load_model(base_path)
            language = "en"
            file = request.files['audio']
            if file and allowed_file(file.filename): 
                to_annotate = os.path.join(app.config["UPLOAD_DIR"], file.filename)
                file.save(to_annotate)

                options = DecodingOptions(
                    task = request.form['task'], 
                    language = request.form['language'],
                    log_tensors = True
                )
                print(f"input options task={options.task}, language={options.language}")
                print(f"log_tensors={options.log_tensors}")

                webm = AudioSegment.from_file(to_annotate, 'webm')
                to_annotate_wav = to_annotate.replace('webm', 'wav')
                webm.export(to_annotate_wav, format='wav') 
                
                mel = log_mel_spec(to_annotate_wav) 
                mel = pad_or_trim(mel, length=2*model.dims.n_audio_ctx) 

                print(f"input audio shape: {mel.shape}")
                result = decode(model, mel, options)
                text, language = result.text, result.language
                print(text, language)
            else:
                text = f"incorrect file format, allowed exts {str(allowed_exts)[1:-1]}"
            del model
            return json.dumps({"text":text, "language":language})
        else:
            return render_template('main.html')
    except:
        print(traceback.format_exc())
        print(f"pid: {os.getpid()}, for file:{to_annotate}")
        del model
        return json.dumps({"text":"__traceback__error__"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', ssl_context='adhoc', port=os.environ.get('PORT', 5000))


