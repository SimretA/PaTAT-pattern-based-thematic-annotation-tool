In two different terminals, do:

# Frontend
- cd into frontend
- npm install --legacy-peer-deps
- run npm start

# Backend 
- open a new terminal
- cd into backend
- create a new environment in conda
- install requirements.txt
- run `python -m spacy download en_core_web_sm`
- uvicorn main:app --port 8080 --host 0.0.0.0
