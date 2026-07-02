# Hybrid Reasoning with LLMs and Logic Programming

Diploma thesis — Department of Electrical and Computer Engineering, 
Hellenic Mediterranean University (HMU).

**Author:** Ilias Tatsios, TH20566
**Supervisor:** Prof. Sotiris Batsakis  
**July 2026**

## Overview

This thesis designs and evaluates three reasoning pipelines 
for logical inference tasks:

- **LLM + Prolog** — declarative hybrid approach
- **LLM + Python** — imperative hybrid approach  
- **LLM Only (CoT)** — Chain-of-Thought baseline

Evaluated on FOLIO (204 problems) and ProofWriter (300 problems) 
using DeepSeek V3.2, GPT-5, and Gemini 2.5 Flash.

## Project Structure
```
├── src/
│   ├── llm_client.py          # LLM API communication
│   ├── pipeline.py            # Three pipeline implementations
│   ├── prolog_engine.py       # SWI-Prolog subprocess execution
│   ├── python_engine.py       # Python subprocess execution
│   ├── data_loader.py         # FOLIO dataset loader
│   ├── proofwriter_loader.py  # ProofWriter dataset loader
│   └── utils.py               # Result extraction, code cleaning
├── evaluate.py                # FOLIO evaluation script
├── evaluate_proofwriter.py    # ProofWriter evaluation script
├── data/                      # Datasets (not included)
└── .env                       # API keys (not included)
```

## Requirements

- Python 3.12+
- SWI-Prolog (installed via Homebrew or apt)
- API keys for DeepSeek, OpenAI, and/or Google Gemini

## Setup

```bash
pip install python-dotenv google-genai openai
```

Create a `.env` file:
```
DEEPSEEK_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

## Usage

```bash
# Run FOLIO evaluation
python evaluate.py

# Run ProofWriter evaluation  
python evaluate_proofwriter.py
```

## Datasets

- [FOLIO](https://github.com/Yale-LILY/FOLIO) — download and place 
  in `data/folio/`
- [ProofWriter](https://allenai.org/data/proofwriter) — download 
  and place in `data/proofwriter/`

## Key Results

| Pipeline | DeepSeek | GPT-5 | Gemini Flash |
|----------|----------|-------|--------------|
| Prolog   | 44.6%    | 72.1% | 69.1%        |
| Python   | 74.5%    | 77.7% | 79.9%        |
| CoT      | 83.3%    | 85.3% | 85.8%        |

## License

This project was developed as part of a diploma thesis at HMU.
