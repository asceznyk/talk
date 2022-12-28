import os
import io
import wave

import numpy as np

from flask import Flask, render_template, request
from pydub import AudioSegment

import talk

app = Flask(__name__)
#app.config["UPLOAD_DIR"] = "uploaded"

@app.route("/", methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        return False ## will put something in here!
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))







