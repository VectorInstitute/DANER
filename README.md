# DANER
Data Annotation Tool for Named Entity Recognition (DANER) using Active Learning and Transfer Learning
- Design Doc: https://docs.google.com/document/d/1zSQbbSxcBhPfIelZi7ElRYUa_8lnOrIdEjPqiBOD-Dk/edit?usp=sharing

# Reproducing

- Environment
  ```bash
  # Run at the project root
  export DANER=$(pwd)
  python3 -m pip install --user --upgrade pip
  python3 -m pip --version
  python3 -m pip install --user virtualenv
  python3 -m venv env
  source env/bin/activate
  pip install -r requirement.txt

  cd $DANER/frontend
  npm install
  ```

- Run DANER on local machine.
  - Backend: `python $DANER/backend/app.py`
  - Frontend
    - Open `$DANER/frontend/dist/spa/index.html` in Browser.
    - Build the frontend: `quasar build`
    - Run in development mode: `quasar dev`

- Run DANER on cluster
  - Backend (Run on cluster):
    - `srun --mem=16G -c 4 --gres=gpu:1 -p interactive --qos=high --pty bash`
    - `python $DANER/backend/app.py`
  - Setup Vector [VPN](https://support.vectorinstitute.ai/Vaughan_SSL_VPN_and_JupyterHub)
  - Frontend (Run on local computer)
    - Open `$DANER/frontend/dist/spa/index.html` in Browser.
    - Build the frontend: `quasar build`
    - Run in development mode: `quasar dev`

- Note
  - You may need to configure the baseURL in the GUI.