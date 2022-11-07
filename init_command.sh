#!/bin/bash

pip install pyserini
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-11-jdk openjdk-11-demo openjdk-11-doc openjdk-11-jre-headless openjdk-11-source
pip install faiss-cpu
export JVM_PATH=/usr/lib/jvm/java-1.11.0-openjdk-amd64/lib/server/libjvm.so