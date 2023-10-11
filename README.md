## LLM Science Exam Competition

This repository contains code for the ??? place solution in the [Kaggle - LLM Science Exam competition](https://www.kaggle.com/competitions/kaggle-llm-science-exam) on Kaggle. The complete solution write-up is [here]().

<h1 align="center">
<img src="./imgs/question_format.png" alt="Model Architecture" width="1000">
</h1>

## Future Improvements

Retrieval/Reranking
- Could have fine-tuned retrieval/re-ranker models to QA with wikipedia data
- Should have focused more on evaluating RAG performance
- Ensembled different retrieval models
- Should have used multiple wiki sources (some pieces missing)

LLM
- Should have experimented with a larger LLM using PEFT (7-13 billion params)
- Ensembled diverse models

CV
- Focus on setting up a correlated CV early in the competition

## Other Attempts

- Prefix-tuning w/ Flan-T5
- SOTA retrieval models (bge-base-en-v1.5, gte-base, all-mpnet-base-v2, etc.)

## Positives

- Collected >7 million English Wikipedia articles
- Enhanced Huggingface `AutoModelForMultipleChoice` class by adding other answers as context
- Sharded FAISS to fit large retrieval model in <13GB of RAM
- Setup good create_data pipeline to test retrieval hyperparameters
