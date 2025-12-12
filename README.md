Here is the final, clean `README.md` content in English, with all citation tags removed, ready for you to copy and paste.

---

# 📖 Memoir for Seniors: Oral History Dialogue and Memoir System

This project aims to transform the unstructured transcripts of seniors' oral histories into structured data and emotionally rich, long-form memoirs by integrating Natural Language Processing (NLP), Large Language Models (LLM), and Affective Computing.

## 🌟 `main` Branch Overview

The `main` branch serves as the **core data and configuration repository** for the project. It contains only the foundational and finalized data files required for the experiments.

**Please Note:** All core code, experimental scripts, and specific processing logic (including the dialogue system modules) reside within the dedicated **`exp1`** through **`exp4`** branches.

### 📁 `main` Branch Data Contents

| Folder | Core Content | Data Source and Purpose |
| :--- | :--- | :--- |
| **`data_json/`** | **Raw/Generated Data (Gold Standard Corpus)** | Contains $\mathbf{50}$ oral history dialogue files. This corpus uses a **Hybrid Strategy** (10 real, anonymized interviews + 40 GPT-4o synthesized transcripts). These files are annotated with the initial multi-layered scheme (structure, emotion, entities). |
| **`data_processed/`** | **Final Structured Input Data** | Contains the $\mathbf{50}$ files after prediction and integration from Experiment 1 (NER) and Experiment 2 (Emotion). This integrated data serves as the **final structured input** for Experiments 3 and 4. |

---

## 🔬 Experimental Goals and Branch Structure

The project validates and optimizes the system's core components through four distinct experimental branches:

| Branch | Experiment Focus | Core Objective and Key Output |
| :--- | :--- | :--- |
| **`exp1`** | **Named Entity Recognition (NER)** | **Goal**: Establish the factual foundation (Knowledge Base). Optimized **RoBERTa-base** using an "Aggressive Optimization Strategy" to transform text into structured entities. |
| **`exp2`** | **Emotion Classification** | **Goal**: Build the affective module. Trained a **Linear SVM** (Macro-F1 of 0.561) on IEMOCAP to generate reliable fine-grained emotion tags for all 50 interviews. |
| **`exp3`** | **Question Generation (QG)** | **Goal**: Quantify the value of structured context on dialogue depth. Conducted a **GPT-4o Ablation Study (ABCD Groups)** to measure the impact of entity and emotion tags on question quality.  |
| **`exp4`** | **Memoir Generation** | **Goal**: Explore long-form narrative application and PII-safe rewriting. Compared four generation strategies (Baseline, RAG, Few-shot, PII-safe) using BERTScore and human ratings. |

---

## 📈 Key Findings Summary (From Final Report)

* **Emotional Context Validation (Exp 3):**
    * The Emotion Focused Group (C) significantly excelled in **Depth** and **Emotional Engagement** (mean score $\mathbf{4.93}$), proving that emotion tags drive more empathetic inquiries.
    * The Full Model Group (D) achieved the highest **Relevance** ($\mathbf{4.96}$), confirming that combining factual (Entities) and affective (Emotions) contexts is the most effective approach for conversational fitness.
* **NER Performance (Exp 1):** The "Aggressive Optimization Strategy" successfully boosted RoBERTa-base **Recall** from $0.52$ to $\mathbf{0.67}$.
* **Memoir Generation Limitations (Exp 4):**
    * The **Baseline** (Zero-shot) approach achieved the highest BERTScore F1 and best human ratings for Coverage, Faithfulness, and Chronology.
    * **Retrieval-Augmented Generation (RAG)** suffered from significant information loss due to retrieval limits, resulting in the lowest BERTScore Recall.
    * **PII-safe** rewriting, while protecting privacy, sacrificed the narrative specificity and factual basis.
