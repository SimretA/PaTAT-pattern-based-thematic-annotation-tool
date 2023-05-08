## Notes:
Currently uses Python 3.9

In two different terminals, do:

# Frontend
- cd into frontend
- npm install --legacy-peer-deps
- run `npm start`

# Backend 
- cd into backend
- create a new environment (for example, using miniconda or python3's venv at the command line, i.e., `python3 -m venv ./venv` and then `venv/bin/activate`)
- `pip install -r requirments0.txt`
- run `python -m spacy download en_core_web_sm`
- `uvicorn main:app --port 8080 --host 0.0.0.0`
