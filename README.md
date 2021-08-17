# DANER Demo
Data Annotation Tool for Named Entity Recognition using Active Learning and Transfer Learning

# Project Structure


# Reproducing

- Environment
```bash
export DANER=$(pwd)
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirement.txt

cd $DANERROOT/frontend
npm install
```

- Running on local machine.

```bash
cd $DANER/frontend
quasar build

cd $DANER/backend
python app.py

# open $DANER/frontend/dist/spa/index.html
```

- Run on cluster

```bash
quasar dev


```


# How to customize?