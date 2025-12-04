import json
import os
from tqdm import tqdm

# ================= 配置区域 =================
RAW_DATA_DIR = "./data_json"       # 原始数据 (含 Interviewer)
KB_DIR = "./knowledge_base"        # 现有 KB (含 Subject 实体)
OUTPUT_DIR = "./knowledge_base_full" # 输出修正后的文件夹
# ===========================================

def main():
    if not os.path.exists(RAW_DATA_DIR) or not os.path.exists(KB_DIR):
        print("❌ 错误: 找不到数据文件夹。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    kb_files = [f for f in os.listdir(KB_DIR) if f.endswith('.json') and f.startswith('KB_')]
    print(f"📄 开始修补 {len(kb_files)} 个文件 (保留原结构 + 插入 Interviewer)...")

    for kb_filename in tqdm(kb_files, desc="Patching"):
        raw_filename = kb_filename.replace("KB_", "")
        kb_path = os.path.join(KB_DIR, kb_filename)
        raw_path = os.path.join(RAW_DATA_DIR, raw_filename)

        if not os.path.exists(raw_path):
            continue

        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 1. 创建 Subject 数据的查找表 (turn_id -> list of sentences)
            # KB 文件里的结构通常是 extracted_knowledge -> [ {turn_id, original_text, predicted_entities}, ... ]
            # 但一个 turn 可能对应多个 entries (如果是按句拆分的)
            kb_lookup = {}
            
            # 检查 KB 里的字段名 (可能是 extracted_knowledge 或 dialogue_analysis)
            source_list = kb_data.get("extracted_knowledge", kb_data.get("dialogue_analysis", []))
            
            for item in source_list:
                tid = item.get("turn_id")
                if tid is not None:
                    if tid not in kb_lookup:
                        kb_lookup[tid] = []
                    kb_lookup[tid].append(item)

            # 2. 构建新的对话流
            new_dialogue_content = []

            # 遍历原始对话，按顺序重建
            for turn in raw_data.get("dialogue_turns", []):
                turn_id = turn.get("turn_id")
                speaker = turn.get("speaker")
                
                # === A. 如果是 Interviewer: 直接插入原始文本 ===
                if speaker == "Interviewer":
                    new_dialogue_content.append({
                        "turn_id": turn_id,
                        "speaker": "Interviewer",
                        "text": turn.get("text", "")
                    })
                
                # === B. 如果是 Subject: 从 KB 里取回原来的实体数据 ===
                elif speaker == "Subject":
                    # 检查 KB 里有没有这个 turn 的数据
                    if turn_id in kb_lookup:
                        # 直接把 KB 里对应这个 turn 的所有 entries 加进去
                        # 这样就保留了原来的分句结构和实体
                        for kb_entry in kb_lookup[turn_id]:
                            # 给它加一个 speaker 标签，保持格式统一
                            kb_entry["speaker"] = "Subject"
                            new_dialogue_content.append(kb_entry)
                    else:
                        # 如果 KB 里没这个 turn (极少见)，就用原始文本兜底
                        text_content = ""
                        if "sentences" in turn:
                             text_content = " ".join([s["text"] for s in turn["sentences"]])
                        new_dialogue_content.append({
                            "turn_id": turn_id,
                            "speaker": "Subject",
                            "original_text": text_content,
                            "predicted_entities": []
                        })

            # 3. 生成最终结构
            final_output = {
                "session_id": kb_data.get("session_id", raw_data.get("session_id")),
                "profile": raw_data.get("profile", {}),
                "dialogue_content": new_dialogue_content
            }

            # 保存
            with open(os.path.join(OUTPUT_DIR, kb_filename), 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 出错 {kb_filename}: {e}")

    print(f"✅ 完成！请检查文件夹: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()