from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# ================= 配置 =================
# 指向您刚才训练好的模型路径
MODEL_PATH = "./ner_model_output/final_model"
# 或者是您预训练后的模型路径 (如果您跑了 DAPT)
# MODEL_PATH = "./bert-memoir-adapted" 
# =======================================

def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        # 1. 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。请确认您已经运行过 train.py 并成功保存了模型。")
        return

    # 2. 创建 NER pipeline (推理管道)
    # aggregation_strategy="simple" 会自动把被切碎的 sub-words (如 'New', 'York') 合并成一个实体
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    print("\n" + "="*50)
    print("🎉 模型加载成功！现在您可以输入英文句子来测试了。")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("="*50 + "\n")

    # 3. 循环输入
    while True:
        text = input("请输入句子 (English): ")
        if text.lower() in ['exit', 'quit']:
            break
        
        if not text.strip():
            continue

        # 4. 进行预测
        results = ner_pipeline(text)

        # 5. 打印结果
        if not results:
            print("  -> 未检测到任何实体。")
        else:
            print(f"\n  [检测结果]:")
            for entity in results:
                # entity 字典包含: entity_group (标签), score (置信度), word (实体词), start/end (位置)
                label = entity['entity_group']
                word = entity['word']
                score = entity['score']
                print(f"   - {word:<20} : {label} ({score:.2%})")
        print("-" * 30)

if __name__ == "__main__":
    main()