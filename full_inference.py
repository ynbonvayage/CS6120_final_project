import os
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm  # 进度条库

# ================= 配置区域 =================
# 1. 模型路径 (请确保指向您 F1 分数最高的那个模型)
MODEL_PATH = "./ner_model_roberta_base/final_model" 

# 2. 数据路径
INPUT_DIR = "./data_json"
OUTPUT_DIR = "./knowledge_base"  # 推理结果将保存在这里

# ===========================================

def main():
    # 1. 检查环境
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型路径 {MODEL_PATH}")
        print("请修改脚本中的 MODEL_PATH 变量，指向您训练好的模型文件夹。")
        return

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到数据文件夹 {INPUT_DIR}")
        return

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已创建输出目录: {OUTPUT_DIR}")

    # 2. 加载模型
    print(f"🚀 正在加载模型: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        # aggregation_strategy="simple" 会自动合并 B- 和 I- 标签 (例如 "New" + "York" -> "New York")
        ner_pipeline = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple",
            device=-1 # 如果有GPU改用 0，没有则用 -1 (CPU)
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 获取文件列表
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"📄 找到 {len(files)} 个文件，开始全量推理...")

    # 4. 循环处理每个文件
    success_count = 0
    
    # 使用 tqdm 显示进度条
    for filename in tqdm(files, desc="Processing Files"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"KB_{filename}") # KB = Knowledge Base

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 准备存储提取结果的结构
            extracted_data = {
                "session_id": data.get("session_id", "unknown"),
                "profile": data.get("profile", {}),
                "extracted_knowledge": [] # 这里存放模型提取出来的实体
            }

            # 遍历对话
            for turn in data.get("dialogue_turns", []):
                # 我们主要关心 Subject (受访者) 的回答
                if turn.get("speaker") == "Subject" and "sentences" in turn:
                    for sent in turn["sentences"]:
                        text = sent.get("text", "")
                        if not text:
                            continue

                        # === 核心步骤：模型推理 ===
                        predictions = ner_pipeline(text)
                        # =======================

                        # 整理预测结果
                        entities = []
                        for pred in predictions:
                            entities.append({
                                "text": pred['word'],
                                "type": pred['entity_group'],
                                "confidence": f"{pred['score']:.4f}" # 保留置信度
                            })

                        # 只有当提取到实体时才保存，保持数据整洁
                        if entities:
                            extracted_data["extracted_knowledge"].append({
                                "turn_id": turn.get("turn_id"),
                                "original_text": text,
                                "predicted_entities": entities
                            })

            # 保存结果文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            success_count += 1

        except Exception as e:
            print(f"\n⚠️ 处理文件 {filename} 时出错: {e}")

    print("\n" + "="*50)
    print(f"✅ 全量推理完成！")
    print(f"📊 成功处理: {success_count}/{len(files)}")
    print(f"📂 结果已保存在: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)

if __name__ == "__main__":
    main()