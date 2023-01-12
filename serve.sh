#!/bin/sh
gcsfuse --implicit-dirs whisper_weights ./assets/
gunicorn --certfile cert.pem --keyfile key.pem --bind :$PORT --workers 9 --threads 8 --timeout 0 home:app



