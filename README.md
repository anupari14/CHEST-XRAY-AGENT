# CHEST-X-RAY-AGENT

<p align="center">

  <!-- Language -->
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/JavaScript-ES6+-yellow?logo=javascript&style=for-the-badge"/>

  <!-- ML -->
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-ee4c2c?logo=pytorch&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/YOLOv8-Detection-black?style=for-the-badge"/>

  <!-- Backend -->
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Uvicorn-ASGI_Server-222222?style=for-the-badge"/>

  <!-- Frontend -->
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&style=for-the-badge"/>

  <!-- LLM -->
  <img src="https://img.shields.io/badge/OpenAI-LLM-412991?logo=openai&style=for-the-badge"/>

  <!-- Vector DB -->
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Search-ff69b4?style=for-the-badge"/>

  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>

</p>

An AI-powered clinical assistant for chest X-ray analysis, featuring automated report generation, semantic search, and a secure, dual-frontend interface. The system uses a YOLOv8 model to detect up to 20 different conditions, generates structured reports using an LLM, and provides a chat-based interface for clinicians to query patient records.

## Features

-   **AI-Powered Chest X-Ray Analysis**: Upload CXR images (PNG, JPG, DICOM) to get predictions for 20 configured conditions using a YOLOv8 model.
-   **Automated Report Generation**: Utilizes a Large Language Model (GPT-4o-mini) to generate comprehensive, structured radiology reports from model findings.
-   **Rich PDF Export**: Creates professional, multi-page PDF reports including patient demographics, input images, AI-generated bounding box overlays, and the full text report.
-   **Patient Management**: A full CRUD API and UI for registering and managing patient records via a simple JSON database.
-   **Semantic Search & RAG**: A sophisticated chatbot and search interface to ask natural language questions across the entire report database, powered by a ChromaDB vector store.
-   **Dual Frontend Options**: Includes both a Streamlit application (`ui.py`) and a modern React application (`medical-agent/`) for interacting with the backend.
-   **Secure API**: A FastAPI backend with simple token-based authentication for all endpoints.

## How It Works

The application follows a multi-stage pipeline from image upload to clinical insight:

1.  **Image Upload**: A user uploads a chest X-ray image and associated patient ID through the web interface.
2.  **AI Inference**: The FastAPI backend uses a YOLOv8 model to perform object detection on the image, identifying potential abnormalities and their locations.
3.  **Report Generation**: The findings from the model are passed to an LLM (e.g., GPT-4o-mini) with a specialized prompt to draft a structured radiology report, including findings and impressions.
4.  **Artifact Creation**: A comprehensive PDF report is generated, bundling patient information, the original CXR, heatmap overlays of the findings, and the AI-generated text report.
5.  **Vector Ingestion**: The text content of the report is chunked, converted to embeddings (using OpenAI or a local model), and stored in a ChromaDB vector database for semantic search.
6.  **Clinical Q&A**: A user can ask natural language questions via the chatbot. The query is used to search the vector database, and the retrieved context is fed to an LLM to generate a synthesized answer (Retrieval-Augmented Generation).

## Getting Started

### Prerequisites

-   Python 3.9+ and `pip`
-   Node.js and `npm` (for the React frontend)
-   An [OpenAI API Key](https://platform.openai.com/account/api-keys) for report generation and embeddings.

### Installation & Running

**1. Clone the repository:**

```sh
git clone https://github.com/anupari14/CHEST-XRAY-AGENT.git
cd CHEST-XRAY-AGENT
```

**2. Backend Setup (FastAPI):**

First, set up and run the Python backend, which serves the API for both frontends.

```sh
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
# Create a .env file in the root directory:
touch .env
```

Add your OpenAI API key to the `.env` file. You can also configure the demo user credentials.

**.env**
```
OPENAI_API_KEY="sk-..."
APP_USERS="demo:demo"
```

**Run the backend server:**

```sh
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

**3. Frontend Setup (React - Recommended):**

The React application provides a modern and responsive user experience.

```sh
# Navigate to the React app directory
cd medical-agent

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The React frontend will be accessible at `http://localhost:5173`. Use the credentials `demo:demo` to log in.

**4. Frontend Setup (Streamlit - Alternative):**

An alternative Streamlit frontend is also available.

```sh
# From the root directory, run the Streamlit app
streamlit run ui.py
```

The Streamlit frontend will open in your browser, likely at `http://localhost:8501`.

## Project Structure

```
├── app.py                  # Main FastAPI application
├── ui.py                   # Streamlit Frontend application
├── requirements.txt        # Python dependencies
├── medical-agent/          # React + Vite Frontend
│   ├── src/
│   └── package.json
├── agents/                 # Agent tools and utilities
│   └── tools.py
├── models/                 # Inference models (YOLOv8)
│   └── yolo.py
├── db/                     # Vector database logic
│   └── vectors.py
├── prompts/                # Prompts for the LLM
│   └── report_prompt.md
├── utils/                  # Helper scripts (PDF export, DICOM conversion)
│   └── pdf_export.py
├── artifacts/              # (Generated) Default location for uploaded files, reports, and DBs
└── vectordb/               # (Generated) Default location for ChromaDB vector store
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

