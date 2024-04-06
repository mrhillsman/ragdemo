### work in progress

### simple steps to run; more detailed readme to follow

- setup a machine that has access to a GPU with at least 24G RAM
- clone this repo
- ensure you have python (3.10) and pip install
- pip install pipenv
- in the repo root directory run pipenv shell
- ensure you are using the python virtual environment
- run pip install to install the dependencies
- run the preprocess.py file to create your embeddings (python preprocess.py)
- run the application (chainlit run app.py)
- chat interface will be available at localhost:8000