# Experiment 1: Named Entity Recognition (NER)

## Overview
This repository contains the implementation for **Experiment 1** of the **Memoir Generation System**. The goal of this module is to extract structured information from unstructured oral history transcripts.

We fine-tuned a **RoBERTa-base** model to recognize **7 specific entity types** essential for generating personalized biographies. The output of this pipeline is a **structured Knowledge Base (JSON)** used by downstream Question Generation and Summarization modules.

---

## Repository Structure
```
├── data_bio/              # Preprocessed data in BIO format (Input for training)
├── data_json/             # Raw annotated JSON files (Source data)
├── dataset_50/            # The full Gold Standard corpus (50 documents)
├── knowledge_base/        # Initial generated Knowledge Base files
├── knowledge_base_full/   # Finalized, post-processed Knowledge Base
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── full_inference.py      # Batch inference script for all 50 documents
├── inference.py           # Single-instance inference script for testing
├── patch_kb.py            # Utility script to update/fix Knowledge Base entries
├── preprocess.py          # Script to convert raw JSON -> BIO format
└── train.py               # Main training script (Fine-tuning RoBERTa)
```

---

## Entity Schema
The model is trained to recognize the following **7 entity types** customized for biographical narratives:

- **PERSON**: Family members, friends, historical figures.
- **LOCATION**: Cities, specific venues, countries.
- **ORGANIZATION**: Workplaces, schools, unions.
- **TIME**: Dates, years, eras (e.g., "the 60s").
- **EVENT**: Historical events (e.g., "Vietnam War") or personal milestones.
- **OCCUPATION**: Job titles, roles (e.g., "Steelworker").
- **ARTIFACT**: Meaningful objects or heirlooms.

---

## Getting Started

### 1. Data Preprocessing
Convert the raw annotated JSON files into the **BIO (Beginning-Inside-Outside)** format required for token-level classification.
```bash
python preprocess.py
```

- **Input**: `data_json/`
- **Output**: Generates training/testing splits in `data_bio/`

### 2. Model Training
Fine-tune the **RoBERTa-base** model on the BIO dataset. This script handles tokenization alignment and metric evaluation.
```bash
python train.py
```

- **Model**: `roberta-base`
- **Optimization**: Uses aggressive learning rate with linear warmup.
- **Output**: Saves the best model weights.

### 3. Inference & Knowledge Base Generation
Run the trained model on the full dataset to generate the structured Knowledge Base.

**For Batch Processing (All 50 docs):**
```bash
python full_inference.py
```

- **Output**: Generates JSON files in `knowledge_base/` containing extracted entities with confidence scores.

**For Single Test Case:**
```bash
python inference.py --input "I worked as a pilot in 1965."
```

### 4. Knowledge Base Maintenance
If structural updates or patches are needed for the generated JSON files, run:
```bash
python patch_kb.py
```

---

## Performance
- **Base Model**: RoBERTa-base
- **Key Advantage**: Superior handling of conversational disfluencies and compound words compared to BERT.
- **Metric**: Optimized for **Recall** to ensure no critical biographical detail is missed.
