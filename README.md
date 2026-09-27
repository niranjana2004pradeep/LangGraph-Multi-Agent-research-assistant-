# 🤖 LangGraph Multi-Agent Research Assistant 
An autonomous AI Deep Research Agent built with **LangGraph**, **LangChain**, and **Tavily API**. 
The system generates a panel of specialized AI analyst personas, conducts parallel web-backed interviews, and synthesizes findings into a cited, comprehensive Markdown report. 
--- 
## 🌟 Key Features 
* **Human-in-the-Loop Oversight**: Pauses execution using LangGraph `interrupt()` so users can review and refine AI analyst personas. 
* **Parallel Fan-Out Execution**: Runs concurrent interview loops for each analyst using LangGraph's `Send` API. 
* **Retrieval-Grounded Q&amp;A**: Uses **Tavily API** for real-time web research to prevent hallucinations. 
* **Structured Pydantic Schemas**: Enforces reliable LLM outputs for personas and search queries. 
* **Map-Reduce Synthesis**: Combines independent analyst sections into a unified report with introduction, conclusion, and citations. 
--- 
## 🚀 Getting Started 
### 1. Clone the repository 
```bash 
git clone https://github.com/YOUR_USERNAME/langgraph-research-agent.git cd langgraph-research-agent 
``` 
### 2. Set up virtual environment &amp; install dependencies 
```bash
python -m venv venv source venv/bin/activate # On Windows: venvScriptsactivate pip install -r requirements.txt ``` 
### 3. Set up Environment Variables Copy `.env.example` to `.env` and fill in your API keys: 
```bash 
cp .env.example .env 
``` 
--- 
## 🏗️ Tech Stack * **Framework**: LangGraph, LangChain * **LLM Providers**: Google Gemini / Groq / OpenAI * **Search Engine**: Tavily API * **Data Validation**: Pydantic