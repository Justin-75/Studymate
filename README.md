## PDF Study Partner MCP Server

An intelligent study companion based on the Groq Cloud LLM (gpt-oss-120b), following the MCP Server specification. Upload a PDF document, and generate summaries, flashcards, and quizzes via natural language queries, with automatic grading.

### Clone Repository

```bash
git clone --recursive https://github.com/your-repo/pdf-study-partner.git
cd studymate
```

- If already cloned but missing submodules:

```bash
git submodule update --init --recursive
```

### Install Dependencies

```powershell
1. Download anaconda
2. Conda create --file environment.yml
3. Conda activate studymate
```

### Configure API Key

1. Get a free API key from [console.groq.com](https://console.groq.com)
2. Create a `.env` file in the project root 

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Start Server

1. Conda activate studymate
2. python server/http_server.py


The server will start at `http://localhost:5000`. Open `test_frontend.html` in your browser to use the application.

### Tools

ingest_pdf 	  - Upload & process PDF|File (multipart/form-data)  Input:  {"doc_id": "..."}     Output: Returns unique doc ID   
generate_material   - Generate material  -   Input: {doc_id, mode, query} Output:        {"data": {...}}     - mode: summary/flashcards/quiz 
grade_quiz	     -   Grade quiz	      Input:    {quiz, answers}    Output:  {score, total, details}  -     quiz is the full object  

### Project Structure
```
your-project/
├── core/                     
├── server/
│   └── http_server.py
├── src/
│   ├── generation/
│   ├── retrieval/
│   ├── llm_client.py          # Groq Cloud API client
│   └── ...
├── tools/
│   └── study_partner_tool.py
├── .env                       # API key (not committed)
├── .gitignore                
├── config.yaml
├── pyproject.toml
├── README.md
├── test_frontend.html
└── test_imports.py
```

### Deployment

1. **Install Python 3.14** 
2. Clone the repo and Install dependency
3. Set your `GROQ_API_KEY` in `.env`
4. Run `python server/http_server.py`
5. Access at `http://localhost:5000`

