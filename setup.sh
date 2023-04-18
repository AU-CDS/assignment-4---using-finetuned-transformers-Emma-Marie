#!/usr/bin/env bash

#create virtual environment
python3 -m venv huggingface_env
source ./huggingface_env/bin/activate

# install hdbscan for BERTopic
sudo apt-get update
sudo apt-get install python3-dev

# requirements
pip install --upgrade pip
pip install --upgrade nbformat
python3 -m pip install -r requirements.txt

#deactivate the venv
deactivate