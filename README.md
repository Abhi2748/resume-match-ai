# Resume Match AI

**Resume Match AI** is a Streamlit-based application that compares a candidate’s resume (PDF) with a job description (TXT), evaluates alignment, highlights strengths and gaps, and suggests job roles with actionable feedback.

---

## Features

- Upload resume (PDF) and job description (TXT)
- Semantic matching using LLMs
- Visual match score with animated UI
- Matching vs. missing skill breakdown
- Personalized job role suggestions with reasoning
- Clean, responsive interface with SVG styling

---

## Project Structure

```
resume-match-ai/
│
├── graphs/
│   └── resume_match_graph.py          # Core LangGraph-based pipeline
│
├── sample_data/
│   ├── sample_resume.pdf
│   └── sample_jd.txt                  # Sample inputs
│
├── utils/
│   ├── data_utils.py
│   ├── embedding_utils.py
│   └── llm_utils.py                   # Helper functions and processing
│
├── app.py                             # Streamlit UI frontend
├── match_engine.py                    # Matching logic module
├── prompts.py                         # Prompt engineering logic
├── blob-scene-haikei.svg              # Custom SVG background
├── test_graph.py                      # CLI test entry
├── requirements.txt                   # Python dependencies
├── .env                               # API keys (excluded in .gitignore)
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resume-match-ai.git
cd resume-match-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create a `.env` file with the following content:

```
OPENAI_API_KEY=your-openai-key
```

### 5. Launch App

```bash
streamlit run app.py
```

---

## Sample Input

Use the files in `sample_data/` to test the system:

- `sample_resume.pdf`
- `sample_jd.txt`

---

## Dependencies

Major dependencies include:

- streamlit
- PyMuPDF
- openai / langchain
- streamlit-extras

Full list in `requirements.txt`.

---

## Contact

mail: abhireddy2748@gmail.com

