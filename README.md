
# Chat Boat API

A Fast API-based service that integrates machine learning models for intelligent chatbot functionality.

## Features

- Fast and lightweight REST API built with FastAPI
- ML model integration for natural language processing
- Easy-to-use endpoints for chat interactions
- Async request handling for improved performance

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

```bash
git clone <repository-url>
cd chat-boat-api
pip install -r requirements.txt
```

## Usage

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
chat-boat-api/
├── main.py
├── requirements.txt
├── models/
├── routes/
└── utils/
```

## API Endpoints

- `POST /chat` - Send a message and receive a response
- `GET /health` - Health check endpoint

## Configuration

Create a `.env` file for environment variables:

```
MODEL_PATH=path/to/model
API_PORT=8000
```

## Contributing

Pull requests are welcome.

## License

MIT
