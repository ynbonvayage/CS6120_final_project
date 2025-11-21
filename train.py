import os
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# ================= 配置 =================
DATA_DIR = "./data_bio"  # 您的数据文件夹
MODEL_CHECKPOINT = "bert-base-cased" # 使用的基础模型
OUTPUT_DIR = "./ner_model_output" # 模型保存路径
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 15 
# =======================================

# 1. 定义标签列表 (必须与 preprocess.py 生成的一致)
LABEL_LIST = [
    "O", 
    "B-PERSON", "I-PERSON",
    "B-LOCATION", "I-LOCATION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-TIME", "I-TIME",
    "B-EVENT", "I-EVENT",
    "B-OCCUPATION", "I-OCCUPATION",
    "B-ARTIFACT", "I-ARTIFACT"
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

def load_custom_dataset(data_files):
    """
    辅助函数：读取 .txt 文件并转换为 Hugging Face Dataset 格式
    """
    def read_conll(filename):
        sentences = []
        labels = []
        if not os.path.exists(filename):
            print(f"Warning: File not found {filename}")
            return {"tokens": [], "ner_tags": []}
            
        with open(filename, 'r', encoding='utf-8') as f:
            current_sent = []
            current_labels = []
            for line in f:
                line = line.strip()
                if not line:
                    if current_sent:
                        sentences.append(current_sent)
                        labels.append(current_labels)
                        current_sent = []
                        current_labels = []
                else:
                    parts = line.split(" ")
                    if len(parts) != 2: continue
                    word, tag = parts[0], parts[1]
                    current_sent.append(word)
                    current_labels.append(label2id.get(tag, 0)) 
            if current_sent:
                sentences.append(current_sent)
                labels.append(current_labels)
        return {"tokens": sentences, "ner_tags": labels}

    train_data = read_conll(data_files["train"])
    val_data = read_conll(data_files["validation"])
    test_data = read_conll(data_files["test"])

    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data),
        "test": Dataset.from_dict(test_data),
    })

def main():
    print("1. 加载数据集...")
    data_files = {
        "train": os.path.join(DATA_DIR, "train.txt"),
        "validation": os.path.join(DATA_DIR, "dev.txt"),
        "test": os.path.join(DATA_DIR, "test.txt"),
    }
    dataset = load_custom_dataset(data_files)

    print("2. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    print("3. 加载模型...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # --- 这里是关键修改点 ---
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch", # 修改了这里：evaluation_strategy -> eval_strategy
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=50,
    )
    # ---------------------

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("4. 开始训练...")
    trainer.train()

    print("5. 在测试集上评估...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\n测试集结果 (Test Results):")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall:    {test_results['eval_recall']:.4f}")
    print(f"F1 Score:  {test_results['eval_f1']:.4f}")

    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"模型已保存至: {os.path.join(OUTPUT_DIR, 'final_model')}")

if __name__ == "__main__":
    main()