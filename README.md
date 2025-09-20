# cooking-chatbot-backend
This project is a Python-based backend for a conversational AI chatbot that suggests meal ideas. It is built with FastAPI and uses LangChain to interact with a locally running fine-tuned Large Language Model via Ollama.

##Prerequisites
Before you begin, ensure you have the following installed:

Python 3.10+

pip (Python package installer)

Ollama

##Setup Instructions

### Clone the Repository

```
git clone https://github.com/williammengyf/cooking-chatbot-backend.git
```

```
cd cooking-chatbot-backend
```


### Create the virtual environment

```
python -m venv venv
```

```
source venv/bin/activate
```


### Install Dependencies

```
pip install -r requirements.txt
```

### Set Up Ollama
Once Ollama is installed and running, pull a model for the chatbot to use.

```
ollama pull qwen3
```

### Configure Environment Variables

Create a file named .env in the root of the project directory. This file will store your configuration.

```
# .env
LLM_MODEL="qwen3"
```

### Running the Application
To start the API server, run the following command from the root directory of the project:

```
uvicorn app.main:app --reload
```

The server will be running on http://127.0.0.1:8000.

### API Usage
The primary endpoint is /chat, which accepts POST requests.
