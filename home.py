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

for fname in os.listdir("/tmp/"):
    print(fname)
    if fname.startswith("talk_user_data_"): app.config["UPLOAD_DIR"] = fname
    else:
        app.config["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="talk_user_data_")
        break

print(app.config["UPLOAD_DIR"])

ALLOWED_EXTS = {'wav', 'mp3', 'ogg'}

model = talk.load_model("assets/tiny.pt")

@app.route("/", methods=['GET', 'POST'])
def main_page():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

    if request.method == 'POST':
        print("post request!")
        if request.form.get('reqtype') == 'upload':
            print(f"reqtype {request.form.get('reqtype')}")
            file = request.files['audio']
            if file.filename == '':
                print("No selected file!") 
            if file and allowed_file(file.filename): 
                to_annotate = os.path.join(app.config["UPLOAD_DIR"], file.filename)
                file.save(to_annotate)
                mel = talk.pad_or_trim(talk.log_mel_spec(to_annotate), length=2*model.dims.n_audio_ctx)
                print(mel.shape) 
                result = model.decode(mel)
                text = result.text
            else:
                text = f"incorrect file format, allowed exts {str(ALLOWED_EXTS)[1:-1]}"

            return json.dumps({"text":text})
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))


