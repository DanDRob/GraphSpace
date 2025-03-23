# GraphSpace Productivity Assistant

GraphSpace is a knowledge graph-based productivity assistant that helps you organize and retrieve information using Graph Neural Networks (GNN) and Retrieval-Augmented Generation (RAG).

## Features

- **Personal Knowledge Graph**: Connect notes, tasks, and contacts in a graph structure
- **AI-Enhanced Retrieval**: Find relevant information using GNN-based embeddings
- **Smart Querying**: Ask natural language questions about your knowledge graph
- **Relationship Discovery**: Automatically find connections between items
- **Document Processing**: Extract knowledge from uploaded documents (PDF, DOCX, TXT)

## Installation

1. Clone this repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the web interface:

```bash
python run_enhanced.py
```

Then open your browser to http://localhost:5000

## Data Structure

Data is stored in JSON format in `data/user_data.json`. The structure consists of:

- `notes`: Collection of notes with content, tags, etc.
- `tasks`: Collection of tasks with title, description, status, etc.
- `contacts`: Collection of contacts with name, email, organization, etc.
- Processed documents are stored in `data/documents/`

## Architecture

- **Knowledge Graph**: Core graph representation using NetworkX
- **GNN Module**: Graph Convolutional Network for learning node embeddings
- **LLM Module**: Retrieval-Augmented Generation for answering queries
- **Document Processing**: Extract text, summaries, topics, and entities from documents

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- Transformers
- FAISS
- Flask (for web interface)
- Document processing libraries (python-docx, PyPDF2, etc.)

## License

MIT
