# 📖 Memoir for Seniors: Oral History Dialogue and Memoir System

This project aims to transform the unstructured transcripts of seniors' oral histories into structured data and emotionally rich, long-form memoirs by integrating Natural Language Processing (NLP), Large Language Models (LLM), and Affective Computing

## 🌟 `main` Branch Overview

The `main` branch serves as the **core data and configuration repository** for the project. It contains only the foundational and finalized data files required for the experiments.

**Please Note:** All core code, experimental scripts, and specific processing logic (including the dialogue system modules) reside within the dedicated **`exp1`** through **`exp4`** branches.

### 📁 `main` Branch Data Contents

| Folder | Core Content | Data Source and Purpose |
| :--- | :--- | :--- |
| **`data_json/`** | **Raw/Generated Data (Gold Standard Corpus)** | [cite_start]Contains $\mathbf{50}$ oral history dialogue files[cite: 9]. [cite_start]This corpus uses a **Hybrid Strategy** [cite: 127] [cite_start](10 real, anonymized interviews + 40 GPT-4o synthesized transcripts [cite: 129, 132]). [cite_start]These files are annotated with the initial multi-layered scheme (structure, emotion, entities)[cite: 139]. |
| **`data_processed/`** | **Final Structured Input Data** | [cite_start]Contains the $\mathbf{50}$ files after prediction and integration from Experiment 1 (NER) and Experiment 2 (Emotion)[cite: 43]. This integrated data serves as the **final structured input** for Experiments 3 and 4. |

---

## 🔬 Experimental Goals and Branch Structure

The project validates and optimizes the system's core components through four distinct experimental branches:

| Branch | Experiment Focus | Core Objective and Key Output |
| :--- | :--- | :--- |
| **`exp1`** | **Named Entity Recognition (NER)** | [cite_start]**Goal**: Establish the factual foundation (Knowledge Base)[cite: 191]. [cite_start]Optimized **RoBERTa-base** using an "Aggressive Optimization Strategy" [cite: 217] [cite_start]to transform text into structured entities[cite: 22, 23]. |
| **`exp2`** | **Emotion Classification** | [cite_start]**Goal**: Build the affective module[cite: 254]. [cite_start]Trained a **Linear SVM** (Macro-F1 of 0.561 [cite: 283, 295][cite_start]) on IEMOCAP [cite: 259] [cite_start]to generate reliable fine-grained emotion tags for all 50 interviews[cite: 43]. |
| **`exp3`** | **Question Generation (QG)** | [cite_start]**Goal**: Quantify the value of structured context on dialogue depth[cite: 45, 50]. [cite_start]Conducted a **GPT-4o Ablation Study (ABCD Groups)** to measure the impact of entity and emotion tags on question quality[cite: 56, 380].  |
| **`exp4`** | **Memoir Generation** | [cite_start]**Goal**: Explore long-form narrative application and PII-safe rewriting[cite: 70]. [cite_start]Compared four generation strategies (Baseline, RAG, Few-shot, PII-safe) [cite: 509] [cite_start]using BERTScore [cite: 82] [cite_start]and human ratings[cite: 533]. |
