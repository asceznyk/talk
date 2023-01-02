#!/bin/sh

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker image remove talktome

docker build -t talktome .
docker run -p 5000:5000 -t talktome --privileged



