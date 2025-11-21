import json
import os
import random
import nltk
from collections import Counter, defaultdict

# 确保下载了分词器
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ================= 配置区域 =================
# 输入：您的JSON文件所在的文件夹路径
INPUT_DIR = "./data_json" 
# 输出：处理好的BIO文件存放路径
OUTPUT_DIR = "./data_bio"

# 我们定义的7大实体标签 (Best Practice_v1)
VALID_LABELS = {
    "PERSON", "LOCATION", "ORGANIZATION", 
    "TIME", "EVENT", "OCCUPATION", "ARTIFACT"
}
# ===========================================

def tokenize(text):
    """使用NLTK进行分词"""
    return nltk.word_tokenize(text)

def match_tokens(sentence_tokens, entity_tokens):
    """在句子token序列中寻找实体token序列的起始位置"""
    slen = len(sentence_tokens)
    elen = len(entity_tokens)
    for i in range(slen - elen + 1):
        if sentence_tokens[i:i+elen] == entity_tokens:
            return i
    return -1

def process_single_file(filepath, stats_counter):
    """处理单个JSON文件，返回句子和对应的BIO标签"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    processed_sentences = []

    # 遍历每一个对话轮次
    for turn in data.get("dialogue_turns", []):
        # 只处理 Subject (受访者) 的回答，或者您也可以包含 Interviewer
        # 这里假设我们主要训练 NER 模型识别回忆录内容，通常 Subject 内容更丰富
        if "sentences" not in turn: 
            continue
            
        for sent_obj in turn["sentences"]:
            text = sent_obj["text"]
            # 1. 分词
            tokens = tokenize(text)
            tags = ["O"] * len(tokens) # 初始化全为 O
            
            # 2. 遍历该句子的实体标注
            entities = sent_obj.get("annotations", {}).get("entities", [])
            
            # 为了防止长实体覆盖短实体（虽然少见），先按长度排序
            entities.sort(key=lambda x: len(x["text"]), reverse=True)

            for ent in entities:
                label = ent["type"].upper() # 统一大写
                
                # 检查标签是否在我们的定义范围内，修正可能的拼写错误
                if label not in VALID_LABELS:
                    # 简单的映射修正逻辑（可根据实际情况扩展）
                    if label == "ORG": label = "ORGANIZATION"
                    elif label == "LOC": label = "LOCATION"
                    elif label == "PER": label = "PERSON"
                    else:
                        # 如果是未定义的标签（如 Date分开标了），跳过或记录
                        continue
                
                ent_text = ent["text"]
                ent_tokens = tokenize(ent_text)
                
                # 3. 在句子中找到实体位置并打标签
                start_idx = match_tokens(tokens, ent_tokens)
                
                if start_idx != -1:
                    # 检查是否已经有标签（避免重叠冲突）
                    is_clean = all(tags[i] == "O" for i in range(start_idx, start_idx + len(ent_tokens)))
                    if is_clean:
                        tags[start_idx] = f"B-{label}"
                        for i in range(1, len(ent_tokens)):
                            tags[start_idx + i] = f"I-{label}"
                        
                        # 统计
                        stats_counter[label] += 1
                    else:
                        # print(f"Warning: Overlap detected in {text} for entity {ent_text}")
                        pass
            
            if tokens:
                processed_sentences.append((tokens, tags))

    return processed_sentences

def save_to_conll(sentences, output_path):
    """将数据保存为标准的 CoNLL 格式 (Token \t Tag)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for tokens, tags in sentences:
            for token, tag in zip(tokens, tags):
                f.write(f"{token} {tag}\n")
            f.write("\n") # 句与句之间空一行

def main():
    # 1. 获取文件列表
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 文件夹 {INPUT_DIR} 不存在。请创建该文件夹并将JSON放入其中。")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"找到 {len(files)} 个 JSON 文件。")
    
    if len(files) == 0:
        return

    # 2. 随机打乱文件顺序 (按文档划分数据集，防止同一人的上下文泄露)
    random.seed(42) # 保证每次运行结果一致
    random.shuffle(files)

    # 3. 划分数据集 (80% Train, 10% Dev, 10% Test)
    # 针对 50 份数据: 40 份训练, 5 份验证, 5 份测试
    n = len(files)
    train_files = files[:int(n*0.8)]
    dev_files = files[int(n*0.8):int(n*0.9)]
    test_files = files[int(n*0.9):]

    print(f"划分计划: 训练集 {len(train_files)}份, 验证集 {len(dev_files)}份, 测试集 {len(test_files)}份")

    # 4. 处理并合并数据
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    global_stats = Counter()

    def process_set(file_list, output_name):
        all_sentences = []
        for filename in file_list:
            path = os.path.join(INPUT_DIR, filename)
            sentences = process_single_file(path, global_stats)
            all_sentences.extend(sentences)
        
        out_path = os.path.join(OUTPUT_DIR, output_name)
        save_to_conll(all_sentences, out_path)
        return len(all_sentences)

    n_train = process_set(train_files, "train.txt")
    n_dev = process_set(dev_files, "dev.txt")
    n_test = process_set(test_files, "test.txt")

    # 5. 输出统计报告
    print("\n" + "="*30)
    print("处理完成！统计报告 (Experiment 1 NER)")
    print("="*30)
    print(f"生成的句子数量:")
    print(f"  Train: {n_train}")
    print(f"  Dev:   {n_dev}")
    print(f"  Test:  {n_test}")
    print(f"  Total: {n_train + n_dev + n_test}")
    
    print("\n实体类型分布 (Label Distribution):")
    for label, count in global_stats.most_common():
        print(f"  {label}: {count}")
    
    print("\n输出文件已保存在:", OUTPUT_DIR)
    print("文件格式示例:")
    print("I O")
    print("lived O")
    print("in O")
    print("New B-LOCATION")
    print("York I-LOCATION")

if __name__ == "__main__":
    main()