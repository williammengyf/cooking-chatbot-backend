# cooking-chatbot-backend
This project is a Python-based backend for a conversational AI chatbot that suggests meal ideas. It is built with FastAPI and uses LangChain to interact with a locally running fine-tuned Large Language Model via Ollama.

## Prerequisites
Before you begin, ensure you have the following installed:

Python 3.10+

pip (Python package installer)

Ollama

## Setup Instructions

### Clone the Repository

```
git clone https://github.com/williammengyf/cooking-chatbot-backend.git
```

```
cd cooking-chatbot-backend
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Pull Model

```
pip install modelscope
modelscope download --model 'FractureSSR/RecipeBot' --local_dir 'pretrained/FractureSSR/RecipeBot'
```

### Install and Set Up Ollama

```
curl -fsSL https://ollama.com/install.sh
ollama serve
```

### Run Models on Ollama
```
ollama create recipebot -f Modelfile
ollama pull nomic-embed-text
```

### Configure Environment Variables

Create a file named .env in the root of the project directory. This file will store your configuration.

```
# .env
FINETUNED_LLM_MODEL="recipebot"
```

### Create the Vector Store (One Time Setup)

Run the data ingestion script to process the dataset file and create the local vector store in a ```./chroma_db``` directory.

```
python ingest_data.py
```

You only need to re-run this script when you update your dataset.

### Running the Application
To start the API server, run the following command from the root directory of the project:

```
uvicorn app.main:app --reload
```

The server will be running on http://127.0.0.1:8000.

### API Usage
The primary endpoint is /chat, which accepts POST requests.
