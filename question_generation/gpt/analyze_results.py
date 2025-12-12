import os 
import json
import statistics
from rouge_score import rouge_scorer
from collections import defaultdict
from openai import OpenAI
import pandas as pd

# ===================== API KEY SETUP =====================
# Paste your OpenAI API KEY here
API_KEY = ""

# Auto-read from environment variable if not set
if not API_KEY or API_KEY.startswith("sk-xxx"):
    API_KEY = os.getenv('OPENAI_API_KEY')

# ========================= Config =========================
INPUT_FILE = "experiment_3_generation_results.json"
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True) 
GROUPS = ['A', 'B', 'C', 'D']
BERT_MODEL_TYPE = 'bert-base-uncased'

# Optional: BERTScore
try:
    from bert_score import score as bert_scorer
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️  bert-score not installed\n")

# LLM evaluation prompt
LLM_EVAL_PROMPT = """You are an expert in evaluating interview questions. Rate the following question on a 1-5 scale.

Context: {context}

Generated Question: {question}

Rate on these 4 dimensions (1=Very Poor, 5=Excellent):
1. Relevance: Does the question fit the conversation context?
2. Depth: Does it elicit detailed responses?
3. Emotional Engagement: Does it evoke emotional memories?
4. Naturalness: Does it sound like a human asking?

Output ONLY valid JSON in this format:
{{"relevance": 5, "depth": 5, "emotional_engagement": 5, "naturalness": 5}}
Do not include any other text."""

# ==================== Text Extraction ====================

def extract_text(generated_obj):
    """Extract text from generated object"""
    if isinstance(generated_obj, str):
        return generated_obj
    
    if isinstance(generated_obj, dict):
        raw = generated_obj.get("raw_output", "")
        
        if raw.startswith("API_ERROR"):
            return ""
        
        if generated_obj.get("parsed") and isinstance(generated_obj["parsed"], dict):
            if "text" in generated_obj["parsed"]:
                return generated_obj["parsed"]["text"]
        
        try:
            clean_raw = raw.strip()
            if clean_raw.startswith("```"):
                lines = clean_raw.split('\n')
                clean_raw = '\n'.join(lines[1:-1] if len(lines) > 2 else lines[1:])
            
            parsed = json.loads(clean_raw)
            if "text" in parsed:
                return parsed["text"]
        except:
            pass
        
        return raw
    
    return ""

# ==================== Metrics Calculation ====================

def calculate_rouge(generated_q, target_gt):
    """Calculate ROUGE scores"""
    if not generated_q or not target_gt:
        return {'rouge1_f1': 0.0, 'rougeL_f1': 0.0}
    
    scores = ROUGE_SCORER.score(target_gt, generated_q)
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure
    }

def calculate_bert_scores(samples):
    """Calculate BERTScore in batch"""
    if not BERT_AVAILABLE:
        return {}
    
    print(f"⚙️  Calculating BERTScore ({len(samples) * len(GROUPS)} samples)...")
    
    all_data = {g: {'gen': [], 'tgt': []} for g in GROUPS}
    
    for sample in samples:
        target = sample['target_question_gt']
        for group in GROUPS:
            gen_obj = sample['generated_questions'][group]
            gen_text = extract_text(gen_obj)
            
            if not gen_text:
                gen_text = target
            
            all_data[group]['gen'].append(gen_text)
            all_data[group]['tgt'].append(target)
    
    results = {}
    for group in GROUPS:
        gen_list = all_data[group]['gen']
        tgt_list = all_data[group]['tgt']
        
        try:
            print(f"   Group {group}...", end=" ", flush=True)
            P, R, F1 = bert_scorer(
                gen_list, tgt_list,
                lang="en",
                model_type=BERT_MODEL_TYPE,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            results[group] = F1.tolist()
        except Exception as e:
            print(f"{e}")
    
    return results

def evaluate_with_llm(client, context, question):
    """Get LLM evaluation scores using GPT-4o"""
    if client is None:
        return None
        
    prompt = LLM_EVAL_PROMPT.format(context=context, question=question)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content.strip()
        scores = json.loads(response_text)
        
        return {
            'relevance': scores.get('relevance', 0),
            'depth': scores.get('depth', 0),
            'emotional_engagement': scores.get('emotional_engagement', 0),
            'naturalness': scores.get('naturalness', 0)
        }
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def batch_llm_evaluation(client, samples):
    """Batch LLM evaluation"""
    if client is None:
        return {g: defaultdict(list) for g in GROUPS}
    
    print(f"\n⚙️  LLM Evaluation ({len(samples) * len(GROUPS)} samples)...")
    
    metrics = {g: defaultdict(list) for g in GROUPS}
    success_count = 0
    
    for idx, sample in enumerate(samples):
        context = sample.get('subject_turn', '')
        
        for group in GROUPS:
            gen_obj = sample['generated_questions'][group]
            gen_text = extract_text(gen_obj)
            
            if not gen_text:
                continue
            
            scores = evaluate_with_llm(client, context, gen_text)
            
            if scores:
                metrics[group]['relevance'].append(scores['relevance'])
                metrics[group]['depth'].append(scores['depth'])
                metrics[group]['emotional_engagement'].append(scores['emotional_engagement'])
                metrics[group]['naturalness'].append(scores['naturalness'])
                success_count += 1
        
        if (idx + 1) % 5 == 0:
            print(f"   Progress: {idx + 1}/{len(samples)} ({success_count} evaluated)")
    
    print(f"   Completed: {success_count} LLM evaluations\n")
    return metrics

def init_llm_client():
    """Initialize OpenAI client"""
    if not API_KEY:
        print("⚠️  No API KEY found. Skipping LLM evaluation.")
        print("   Set API_KEY at the top of this script or use environment variable")
        return None
    
    try:
        return OpenAI(api_key=API_KEY)
    except Exception as e:
        print(f" Failed to initialize LLM: {e}")
        return None

# ==================== Table Generation ====================

def create_auto_metrics_table(auto_metrics):
    """Create automatic metrics table"""
    data = []
    for group in GROUPS:
        m = auto_metrics[group]
        
        r1 = statistics.mean(m['rouge1']) if m['rouge1'] else 0
        r1_std = statistics.stdev(m['rouge1']) if len(m['rouge1']) > 1 else 0
        
        rL = statistics.mean(m['rougeL']) if m['rougeL'] else 0
        rL_std = statistics.stdev(m['rougeL']) if len(m['rougeL']) > 1 else 0
        
        if 'bert' in m and m['bert']:
            bert = statistics.mean(m['bert'])
            bert_std = statistics.stdev(m['bert']) if len(m['bert']) > 1 else 0
        else:
            bert = 0
            bert_std = 0
        
        data.append({
            'Group': group,
            'ROUGE-1': f"{r1:.4f}±{r1_std:.4f}",
            'ROUGE-L': f"{rL:.4f}±{rL_std:.4f}",
            'BERTScore': f"{bert:.4f}±{bert_std:.4f}" if bert > 0 else "N/A"
        })
    
    return pd.DataFrame(data)

def create_llm_metrics_table(llm_metrics):
    """Create LLM metrics table"""
    data = []
    for group in GROUPS:
        m = llm_metrics[group]
        
        rel = statistics.mean(m['relevance']) if m['relevance'] else 0
        rel_std = statistics.stdev(m['relevance']) if len(m['relevance']) > 1 else 0
        
        dep = statistics.mean(m['depth']) if m['depth'] else 0
        dep_std = statistics.stdev(m['depth']) if len(m['depth']) > 1 else 0
        
        emo = statistics.mean(m['emotional_engagement']) if m['emotional_engagement'] else 0
        emo_std = statistics.stdev(m['emotional_engagement']) if len(m['emotional_engagement']) > 1 else 0
        
        nat = statistics.mean(m['naturalness']) if m['naturalness'] else 0
        nat_std = statistics.stdev(m['naturalness']) if len(m['naturalness']) > 1 else 0
        
        data.append({
            'Group': group,
            'Relevance': f"{rel:.2f}±{rel_std:.2f}" if rel > 0 else "N/A",
            'Depth': f"{dep:.2f}±{dep_std:.2f}" if dep > 0 else "N/A",
            'Emotional': f"{emo:.2f}±{emo_std:.2f}" if emo > 0 else "N/A",
            'Naturalness': f"{nat:.2f}±{nat_std:.2f}" if nat > 0 else "N/A"
        })
    
    return pd.DataFrame(data)

def create_comparison_table(auto_metrics, llm_metrics):
    """Create comparison table (BCD vs A)"""
    comparison_data = []
    
    # Automatic metrics comparison
    for metric_name in ['rouge1', 'rougeL']:
        a_mean = statistics.mean(auto_metrics['A'][metric_name]) if auto_metrics['A'][metric_name] else 0
        
        for group in ['B', 'C', 'D']:
            g_mean = statistics.mean(auto_metrics[group][metric_name]) if auto_metrics[group][metric_name] else 0
            diff = g_mean - a_mean
            improvement = (diff / a_mean * 100) if a_mean > 0 else 0
            symbol = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "→")
            
            comparison_data.append({
                'Metric': metric_name.upper(),
                'Group': group,
                'Score': f"{g_mean:.4f}",
                'vs_A': f"{symbol}{abs(improvement):.1f}%"
            })
    
    # LLM metrics comparison
    if any(any(llm_metrics[g].values()) for g in GROUPS):
        for metric_name in ['relevance', 'depth', 'emotional_engagement', 'naturalness']:
            a_mean = statistics.mean(llm_metrics['A'][metric_name]) if llm_metrics['A'][metric_name] else 0
            
            if a_mean == 0:
                continue
            
            for group in ['B', 'C', 'D']:
                g_mean = statistics.mean(llm_metrics[group][metric_name]) if llm_metrics[group][metric_name] else 0
                diff = g_mean - a_mean
                improvement = (diff / a_mean * 100) if a_mean > 0 else 0
                symbol = "↑" if diff > 0.1 else ("↓" if diff < -0.1 else "→")
                
                comparison_data.append({
                    'Metric': metric_name.upper(),
                    'Group': group,
                    'Score': f"{g_mean:.2f}",
                    'vs_A': f"{symbol}{abs(improvement):.1f}%"
                })
    
    return pd.DataFrame(comparison_data)

# ==================== Analysis ====================

def analyze_results():
    """Main analysis function"""
    print("=" * 80)
    print("🚀 Quantitative & LLM Evaluation (GPT-4o)")
    print("=" * 80)
    
    if not os.path.exists(INPUT_FILE):
        print(f" File not found: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, 'r') as f:
        samples = json.load(f)
    
    print(f" Loaded {len(samples)} samples\n")
    
    # Initialize automatic metrics
    auto_metrics = {g: defaultdict(list) for g in GROUPS}
    errors = {g: 0 for g in GROUPS}
    
    # Calculate ROUGE
    print("⚙️  Calculating ROUGE...")
    for sample in samples:
        target = sample['target_question_gt']
        
        for group in GROUPS:
            gen_obj = sample['generated_questions'][group]
            gen_text = extract_text(gen_obj)
            
            if not gen_text:
                errors[group] += 1
                continue
            
            rouge = calculate_rouge(gen_text, target)
            auto_metrics[group]['rouge1'].append(rouge['rouge1_f1'])
            auto_metrics[group]['rougeL'].append(rouge['rougeL_f1'])
    
    print("✅ ROUGE completed\n")
    
    # Calculate BERTScore
    bert_scores = calculate_bert_scores(samples)
    for group in GROUPS:
        if group in bert_scores:
            auto_metrics[group]['bert'] = bert_scores[group]
    
    # Initialize LLM client
    client = init_llm_client()
    
    # LLM Evaluation
    llm_metrics = batch_llm_evaluation(client, samples)
    
    # ==================== Output Reports ====================
    
    print("\n" + "=" * 80)
    print(" AUTOMATIC METRICS REPORT")
    print("=" * 80)
    print(f"Total samples: {len(samples)}")
    if any(errors.values()):
        print(f"Errors: {dict(errors)}")
    print("=" * 80 + "\n")
    
    # Automatic metrics table
    auto_table = create_auto_metrics_table(auto_metrics)
    print("AUTOMATIC METRICS (Mean ± Std Dev):")
    print(auto_table.to_string(index=False))
    print()
    
    # LLM metrics table
    print("\n" + "=" * 80)
    print(" LLM EVALUATION METRICS REPORT (GPT-4o)")
    print("=" * 80)
    print("Scale: 1=Very Poor, 5=Excellent\n")
    
    llm_table = create_llm_metrics_table(llm_metrics)
    print("LLM METRICS (Mean ± Std Dev):")
    print(llm_table.to_string(index=False))
    print()
    
    # Best performance
    print("\n" + "=" * 80)
    print(" BEST PERFORMANCE:\n")
    
    best_r1 = max(GROUPS, key=lambda g: statistics.mean(auto_metrics[g]['rouge1']) if auto_metrics[g]['rouge1'] else 0)
    best_rL = max(GROUPS, key=lambda g: statistics.mean(auto_metrics[g]['rougeL']) if auto_metrics[g]['rougeL'] else 0)
    
    print(f"ROUGE-1: Group {best_r1} ({statistics.mean(auto_metrics[best_r1]['rouge1']):.4f})")
    print(f"ROUGE-L: Group {best_rL} ({statistics.mean(auto_metrics[best_rL]['rougeL']):.4f})")
    
    if BERT_AVAILABLE:
        valid_bert = {g: statistics.mean(auto_metrics[g]['bert']) for g in GROUPS if 'bert' in auto_metrics[g] and auto_metrics[g]['bert']}
        if valid_bert:
            best_bert = max(valid_bert, key=valid_bert.get)
            print(f"BERTScore: Group {best_bert} ({valid_bert[best_bert]:.4f})")
    
    # LLM metrics best
    if any(any(llm_metrics[g].values()) for g in GROUPS):
        best_rel = max(GROUPS, key=lambda g: statistics.mean(llm_metrics[g]['relevance']) if llm_metrics[g]['relevance'] else 0)
        best_dep = max(GROUPS, key=lambda g: statistics.mean(llm_metrics[g]['depth']) if llm_metrics[g]['depth'] else 0)
        best_emo = max(GROUPS, key=lambda g: statistics.mean(llm_metrics[g]['emotional_engagement']) if llm_metrics[g]['emotional_engagement'] else 0)
        best_nat = max(GROUPS, key=lambda g: statistics.mean(llm_metrics[g]['naturalness']) if llm_metrics[g]['naturalness'] else 0)
        
        print(f"\nRelevance: Group {best_rel} ({statistics.mean(llm_metrics[best_rel]['relevance']):.2f})")
        print(f"Depth: Group {best_dep} ({statistics.mean(llm_metrics[best_dep]['depth']):.2f})")
        print(f"Emotional Engagement: Group {best_emo} ({statistics.mean(llm_metrics[best_emo]['emotional_engagement']):.2f})")
        print(f"Naturalness: Group {best_nat} ({statistics.mean(llm_metrics[best_nat]['naturalness']):.2f})")
    
    # ==================== Comparison Analysis ====================
    
    print("\n" + "=" * 80)
    print("📈 COMPARISON ANALYSIS (BCD vs Baseline A):\n")
    
    comparison_table = create_comparison_table(auto_metrics, llm_metrics)
    print("COMPARISON TABLE (vs Group A):")
    print(comparison_table.to_string(index=False))
    
    print("\n" + "=" * 80)
    
    # Save results
    output = {
        'automatic_metrics': {
            group: {
                'rouge1': statistics.mean(auto_metrics[group]['rouge1']) if auto_metrics[group]['rouge1'] else 0,
                'rougeL': statistics.mean(auto_metrics[group]['rougeL']) if auto_metrics[group]['rougeL'] else 0,
                'bert': statistics.mean(auto_metrics[group]['bert']) if 'bert' in auto_metrics[group] and auto_metrics[group]['bert'] else None,
                'errors': errors[group]
            } for group in GROUPS
        },
        'llm_metrics': {
            group: {
                'relevance': statistics.mean(llm_metrics[group]['relevance']) if llm_metrics[group]['relevance'] else 0,
                'depth': statistics.mean(llm_metrics[group]['depth']) if llm_metrics[group]['depth'] else 0,
                'emotional_engagement': statistics.mean(llm_metrics[group]['emotional_engagement']) if llm_metrics[group]['emotional_engagement'] else 0,
                'naturalness': statistics.mean(llm_metrics[group]['naturalness']) if llm_metrics[group]['naturalness'] else 0,
            } for group in GROUPS
        },
        'total_samples': len(samples),
        'llm_model': 'gpt-4o'
    }
    
    try:
        with open('experiment_3_full_evaluation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        print("\n Saved: experiment_3_full_evaluation_results.json")
        
        # Save tables as CSV
        auto_table.to_csv('automatic_metrics_table.csv', index=False)
        print(" Saved: automatic_metrics_table.csv")
        
        llm_table.to_csv('llm_metrics_table.csv', index=False)
        print(" Saved: llm_metrics_table.csv")
        
        comparison_table.to_csv('comparison_analysis_table.csv', index=False)
        print(" Saved: comparison_analysis_table.csv")
    except OSError:
        print("\n  Cannot save results (read-only directory)")

if __name__ == "__main__":
    try:
        analyze_results()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()