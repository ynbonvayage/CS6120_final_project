import os
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型路径 (指向您最好的模型)
MODEL_PATH = "./ner_model_roberta_base/final_model" 

# 2. 数据路径
INPUT_DIR = "./data_json"
OUTPUT_DIR = "./knowledge_base"
# ===========================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型路径 {MODEL_PATH}")
        return

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到数据文件夹 {INPUT_DIR}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 正在加载模型: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        ner_pipeline = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple",
            device=-1 
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"📄 开始处理 {len(files)} 个文件 (保留完整对话流)...")

    for filename in tqdm(files, desc="Processing"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"KB_{filename}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 新的输出结构：包含完整的对话流
            kb_output = {
                "session_id": data.get("session_id", "unknown"),
                "profile": data.get("profile", {}),
                "dialogue_analysis": [] 
            }

            for turn in data.get("dialogue_turns", []):
                speaker = turn.get("speaker")
                turn_id = turn.get("turn_id")
                
                # 容器：用于存储这一轮的分析结果
                turn_data = {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "text_content": "" # 稍后填充
                }

                # === 情况 A: 采访者 (只保留文本，不做 NER) ===
                if speaker == "Interviewer":
                    turn_data["text_content"] = turn.get("text", "")
                    # 不加 "entities" 字段，或者留空
                
                # === 情况 B: 受访者 (保留文本 + 做 NER) ===
                elif speaker == "Subject":
                    # 获取句子列表 (兼容新旧格式)
                    sentences = turn.get("sentences", turn.get("sentence_annotations", []))
                    
                    full_text = ""
                    extracted_entities = []

                    for sent in sentences:
                        text = sent.get("text", "")
                        if not text: continue
                        
                        full_text += text + " " # 拼接完整回答以便阅读
                        
                        # --- 模型推理 ---
                        predictions = ner_pipeline(text)
                        for pred in predictions:
                            extracted_entities.append({
                                "text": pred['word'],
                                "type": pred['entity_group'],
                                "confidence": f"{pred['score']:.4f}"
                            })
                        # ----------------
                    
                    turn_data["text_content"] = full_text.strip()
                    turn_data["extracted_entities"] = extracted_entities

                # 将处理好的一轮对话加入结果
                kb_output["dialogue_analysis"].append(turn_data)

            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(kb_output, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}: {e}")

    print(f"✅ 完成！结果已保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()