# Website RAG Chatbot

A local AI-powered chatbot that answers questions using content crawled from a specified website. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant website information and generate context-aware responses.

## Overview

The application crawls website pages, extracts their content, preprocesses the extracted text, and stores vector embeddings in Qdrant. When a user asks a question, the application retrieves relevant content and passes it to a locally running language model to generate an answer.

The chatbot is designed to answer questions based on indexed website content rather than relying on the language model's general knowledge.

## Features

- **Website Crawling:** Crawl a specified website and discover linked pages using Crawl4AI.
- **Content Extraction:** Extract page content in Markdown format.
- **Text Preprocessing:** Split extracted content into manageable chunks with configurable overlap.
- **Duplicate Removal:** Identify identical text chunks across pages to reduce redundant storage while retaining their source references.
- **Semantic Search:** Generate text embeddings using Sentence Transformers and retrieve relevant content using Qdrant.
- **Context-Aware Responses:** Generate answers using retrieved website content and a locally running LLM.
- **Source References:** Associate answers with the website pages used as supporting evidence.
- **Configurable Pipeline:** Set crawling limits, chunk sizes, retrieval parameters, and model configuration through environment variables.
- **Interactive Interface:** Use a Streamlit web interface to index website content and ask questions.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Web Crawling | Crawl4AI |
| Text Processing | Python |
| Embedding Model | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector Database | Qdrant Cloud |
| Large Language Model | Ollama |
| Configuration | Python dotenv |

## Architecture

```text
                 Specified Website
                         |
                         v
                  Crawl4AI Crawler
                         |
                         v
                  Content Extraction
                         |
                         v
                  Text Preprocessing
             (Cleaning, Chunking, Deduplication)
                         |
                         v
                  Embedding Generation
                         |
                         v
                    Qdrant Cloud
                  Vector Collection
                         |
                         |
User Question ------------+
       |
       v
  Query Embedding
       |
       v
  Semantic Retrieval
       |
       v
  Relevance Filtering
       |
       v
  Retrieved Website Context
       |
       v
  Local LLM via Ollama
       |
       v
  Answer with Source References
```

## How It Works

### 1. Website Crawling

Crawl4AI visits the configured starting URL, discovers eligible internal links, and extracts page content. The crawling process is limited by the configured maximum number of pages.

### 2. Content Preprocessing

The extracted Markdown content is processed into smaller text chunks. Chunking allows the application to retrieve relevant sections instead of passing entire web pages to the language model.

Identical normalized chunks can be deduplicated to reduce redundant indexing while preserving references to the pages where the content appears.

### 3. Embedding Generation and Storage

A Sentence Transformers model converts text chunks into numerical vectors. These vectors and their associated text and source metadata are stored in a Qdrant collection.

### 4. Question Retrieval

When a user submits a question, the application generates a query embedding and searches Qdrant for semantically similar chunks. A relevance threshold filters out results that are not sufficiently relevant.

### 5. Answer Generation

The retrieved content is provided to a local language model through Ollama. The model is instructed to answer using the supplied website evidence and avoid inventing unsupported information.

If relevant evidence cannot be found, the application returns a fallback response rather than presenting an unsupported answer.

## Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Ollama installed and running locally.
- A Qdrant Cloud account and a configured vector database.
- Internet access to crawl the target website and connect to Qdrant Cloud.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anugrahaantoo/website-rag-assistant.git
cd website-rag-chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama

Pull the chat model you intend to use. For example:

```bash
ollama pull qwen2.5-coder:3b
```

Make sure Ollama is running locally.

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
WEBSITE_URL=https://your-website.com

QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=website_knowledge

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:3b

MAX_PAGES=30
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
MIN_SCORE=0.35
REQUEST_TIMEOUT=20
```

Replace the example website and Qdrant credentials with your actual configuration. Adjust the model names and parameters to match the models and settings supported by your application.

**Important:** Keep `.env` private. Do not commit API keys or credentials to GitHub.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open the local address displayed in your terminal to access the chatbot.

## Usage

1. Start the application.
2. Configure the target website and crawling settings.
3. Run the website indexing process.
4. Wait for the pages to be crawled, processed, embedded, and stored in Qdrant.
5. Enter a question about the indexed website.
6. Review the generated answer and its source references.

Questions outside the indexed website's content may receive a fallback response.

## Configuration

| Variable | Description |
|---|---|
| `WEBSITE_URL` | Starting URL for website crawling |
| `MAX_PAGES` | Maximum number of pages to crawl |
| `CHUNK_SIZE` | Maximum target size of each text chunk |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | Model used to generate text embeddings |
| `QDRANT_URL` | Qdrant Cloud endpoint |
| `QDRANT_COLLECTION` | Name of the vector collection |
| `OLLAMA_URL` | Local Ollama API endpoint |
| `OLLAMA_MODEL` | Language model used to generate answers |
| `TOP_K` | Maximum number of retrieved chunks considered |
| `MIN_SCORE` | Minimum retrieval score accepted by the application |
| `REQUEST_TIMEOUT` | Configured request timeout |

The retrieval score and threshold must be interpreted according to the similarity metric used by the application.

## Limitations

- Answer quality depends on the completeness and accuracy of the crawled website content.
- Pages that require authentication or use unsupported rendering techniques may not be extracted correctly.
- Crawling is subject to website availability, access restrictions, and the configured page limit.
- Semantic retrieval may return irrelevant content, particularly when questions are vague.
- The chatbot can fail to answer a question even when related information exists if retrieval or relevance filtering does not identify it.
- Response quality and speed depend on the local language model and available hardware.
- Qdrant Cloud requires an internet connection.

## Future Improvements

- Improve retrieval using hybrid search and reranking.
- Add automated evaluation for retrieval accuracy and answer quality.
- Improve handling of dynamically rendered websites.
- Add incremental indexing to avoid rebuilding the entire collection after every crawl.
- Strengthen prompt-injection testing for untrusted website content.
- Add logging, error handling, and performance monitoring.

## License

Add a license if you intend to distribute the project for reuse.