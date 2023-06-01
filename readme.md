# PaTAT
# Overview

- [To build and run from source-code](#to-build-and-run-from-source-code)
  - [Frontend](#frontend)
  - [Backend](#backend)
- [To run in docker](#to-run-in-docker)
  - [Backend](#backend-1)
  - [Frontend](#frontend-1)
- [To run in docker-compose](#to-run-in-docker-compose)
- [To view API docs](#to-view-api-docs)

## Notes:

Currently uses Python 3.9

# To build and run from source-code

In two different terminals, do:

## Frontend

- cd into frontend
- `npm install --legacy-peer-deps`
- `npm start`

## Backend

- cd into backend
- create a new environment (for example, using miniconda or python3's venv at the command line, i.e., `python3 -m venv ./venv` and then `venv/bin/activate`)
- `pip install -r requirements.txt`
- `python -m spacy download en_core_web_sm`
- `uvicorn main:app --port 8080 --host 0.0.0.0`

# To run in docker

In two different terminals, do:

## Backend

- cd into backend
- `docker build -t patat-backend .`
- `docker run -p 8080:8080 patat-backend`

## Frontend

- cd into frontend
- `docker build -t patat-frontend .`
- `docker run -p 3000:3000 patat-frontend`

# To run in docker-compose

- install `docker-compose` if you don't have it installed
- `docker-compose up --build`

# To view API docs

- head to `http://localhost:8080/docs` after running the backend

# To run in AWS instance
- create an instance with GPU: Instance should have at least 30GBs of local storage
- download the generated keypair and `chmod 400 <FILE.pem>` to make it read-only
- launch instance if its never been launched; use instance state drop-down to restart a stopped instance if its already been launched by stopped.
- copy the files into the instance (either scp the directory from your local machine `scp -i "FILE.pem" -r <DIR> ec2-user@INSTANCE_ADDRESS:` or git clone after sshing into the instance `ssh -i "FILE.pem" ec2-user@INSTANCE_ADDRESS`) where INSTANCE_ADDRESS is the public ipv4 DNS address.
- install docker and docker compose: helpful resources ( to install docker [link](https://docs.docker.com/engine/install/) , to install docker-compose [link](https://docs.docker.com/compose/install/linux/), to start docker demeaon [link](https://docs.docker.com/config/daemon/start/)
- Using the Instance UI in Amazon AWS, allow ports 3000 and 8000 to be accessible from your instance inbound and outbound security rules
![Screenshot_6_1_23__3_12_PM](https://github.com/SimretA/PaTAT-pattern-based-thematic-annotation-tool/assets/2320194/56db377e-bf44-42c4-9fac-d3f4fdbc65a9)
- ssh in and go to frontend/src/assets/base_url.jsx and replace the ip address with your instances public ipv4
- follow the running instructions above to run PaTAT



# Reference
If you use our tools/code for your work, please cite the following paper:

Simret Araya Gebreegziabher, Zheng Zhang, Xiaohang Tang, Yihao Meng, Elena L. Glassman, and Toby Jia-Jun Li. 2023. PaTAT: Human-AI Collaborative Qualitative Coding with Explainable Interactive Rule Synthesis. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23). https://doi.org/10.1145/3544548.3581352


