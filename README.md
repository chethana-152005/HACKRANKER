Multi-Domain Support Triage Agent
A terminal-based AI support triage agent that automatically classifies, assesses risk, and generates safe responses for support tickets across three ecosystems: HackerRank, Claude, and Visa.

🎯 Overview
This agent processes support tickets and performs:

Domain Detection: Automatically identifies if a ticket belongs to HackerRank, Claude, or Visa.
Request Classification: Categorizes tickets as product_issue, feature_request, bug, or invalid.
Risk Assessment: Evaluates the safety and sensitivity of each ticket.
Smart Escalation: Escalates high-risk issues (fraud, legal, security, billing disputes) to human agents.
Grounded Response Generation: Creates user-facing responses based ONLY on the provided support corpus.
Fallback Mode: Gracefully degrades to document-based answers when AI quotas are exceeded.
✨ Features
Feature	Description
Semantic Retrieval	Uses FAISS + Sentence Transformers to find the most relevant support documents.
Multi-Domain Support	Handles tickets for HackerRank, Claude, and Visa.
Safety Checks	Detects malicious prompts, fraud keywords, and sensitive topics.
Auto-Escalation	Routes high-risk tickets to human support automatically.
Terminal-Based	Simple CLI interface for processing CSV files.
Graceful Degradation	Falls back to corpus-based answers when AI is unavailable.
🏗️ Project Structure
project/
│
├── data/
│ ├── corpus/
│ │ ├── hackerrank/ # HackerRank support documents
│ │ ├── claude/ # Claude support documents
│ │ └── visa/ # Visa support documents
│ ├── support_tickets.csv # Input tickets to process
│ └── sample_support_tickets.csv
│
├── src/
│ ├── main.py # Entry point
│ ├── triage_agent.py # Core agent logic
│ ├── retriever.py # Semantic search
│ ├── classifier.py # Request classification
│ ├── safety.py # Safety & escalation logic
│ ├── ingest.py # Corpus loading
│ └── prompts.py # LLM prompts
│
├── output/
│ └── predictions.csv # Generated results
│
├── logs/
│ └── log.txt # Execution log
│
├── requirements.txt
├── README.md
└── .env # API keys (not included in submission)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Set API Key
Create a .env file in the project root:
3. Run the Agent
python src/main.py --input data/support_tickets.csv --output output/predictions.csv
4. Check Results
# View output
cat output/predictions.csv

# View logs
cat logs/log.txt
📊 Output Format
The agent generates a CSV file with the following columns:

Column
Description
issue	Original ticket issue text
subject	Ticket subject line
company	Detected company (HackerRank, Claude, Visa, None)
response	Generated response (AI or fallback)
product_area	Classified product area (e.g., Assessments, Billing, Fraud & Disputes)
status	replied or escalated
request_type	product_issue, feature_request, bug, or invalid
justification	Reasoning for the decision

🔧 How It Works
Ticket Processing Pipeline
┌─────────────────┐
│  Input Ticket   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Classify      │  → Detect domain & request type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Safety Check    │  → Assess risk, detect sensitive content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Escalate?      │  → High risk? → Escalate to human
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ Retrieve Docs   │  → Semantic search over corpus
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate Reply  │  → LLM or fallback response
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Output    │
└─────────────────┘
Escalation Rules
The agent automatically escalates tickets involving:

🚨 Fraud or unauthorized charges
⚖️ Legal threats or lawsuits
🔐 Account compromise or security vulnerabilities
💳 Billing disputes and refund requests
🆔 Identity verification issues
⚠️ Any ticket where the user explicitly requests a human agent
🧠 Technology Stack
Component
Technology
LLM	Google Gemini 2.5 Flash
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Vector Search	FAISS
Language	Python 3.10+
CLI	argparse

📝 Example Usage
Input (support_tickets.csv)
Issue,Subject,Company
"My identity has been stolen, what should I do?",Identity Theft,Visa
"I completed a test but my score is wrong",Test Score Dispute,HackerRank
Output (predictions.csv)
issue,subject,company,response,product_area,status,request_type,justification
"My identity has been stolen...","Identity Theft",Visa,"This issue has been escalated...",Fraud & Disputes,escalated,product_issue,High-risk issue requires human review
"I completed a test but...","Test Score Dispute",HackerRank,"This issue has been escalated...",Assessments,escalated,product_issue,Score disputes require human review
🛡️ Safety Features
Prompt Injection Detection: Ignores malicious instructions embedded in tickets.
Sensitive Topic Handling: Automatically escalates PII, fraud, and legal matters.
Response Validation: Checks generated responses for safety before sending.
Corpus Grounding: Only uses provided support documentation—no external knowledge.
google-generativeai>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
python-dotenv>=1.0.0

📄 License
This project is submitted for the HackerRank Orchestrate Hackathon (May 2026).

👤 Author
Hackathon Team Submission

🙏 Acknowledgments
Google Gemini for LLM capabilities
HuggingFace for sentence transformers
FAISS for efficient similarity search

