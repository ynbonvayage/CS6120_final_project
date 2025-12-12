import os
import json
import copy

# ========================= Configuration Area =========================
METADATA_FILE = "experiment_3_metadata.json"
DATA_DIR = "../data_processed"
OUTPUT_FILE = "experiment_3_prompts_input.json"

# ==================== Feature Filtering Helper Functions ====================

def filter_annotations(original_turn_data, group_type):
    """
    Filter annotations in Subject Turn based on experimental group (A, B, C, D).
    A: Only keep life_stage and event (exclude entities and emotions)
    B: Keep life_stage, event, entities
    C: Keep life_stage, event, emotions
    D: Keep all (life_stage, event, entities, emotions)
    """
    filtered_turn = copy.deepcopy(original_turn_data)
    
    if filtered_turn.get("speaker") != "Subject":
        return filtered_turn

    keep_entities = group_type in ['B', 'D']
    keep_emotions = group_type in ['C', 'D']

    for sentence in filtered_turn.get("sentences", []):
        annotations = sentence.get("annotations", {})
        
        # Always keep life_stage and event
        new_annotations = {
            k: v for k, v in annotations.items() if k in ['life_stage', 'event']
        }

        if keep_entities and 'entities' in annotations:
            new_annotations['entities'] = annotations['entities']
            
        if keep_emotions and 'emotions' in annotations:
            new_annotations['emotions'] = annotations['emotions']
            
        sentence["annotations"] = new_annotations
        
    return filtered_turn


def format_turn_for_history(turn, group_type):
    """
    Format a single turn as history text.
    Interviewer: Plain text
    Subject: JSON format (with filtered annotations)
    """
    speaker = turn["speaker"]
    
    if speaker == "Interviewer":
        text = turn.get("text", "")
        return f'Turn {turn["turn_id"]} (Interviewer): "{text}"'
    
    else:  # Subject
        filtered_turn = filter_annotations(turn, group_type)
        sentences = filtered_turn.get("sentences", [])
        sentences_json = json.dumps(sentences, indent=2, ensure_ascii=False)
        return f'Turn {turn["turn_id"]} (Subject):\n{sentences_json}'


def get_complete_dialogue_history(all_turns, end_turn_id, group_type):
    """
    Get complete dialogue history from Turn 1 to end_turn_id.
    """
    history_parts = []
    
    for turn in all_turns:
        if turn["turn_id"] > end_turn_id:
            break
        history_parts.append(format_turn_for_history(turn, group_type))
    
    return "\n\n".join(history_parts)


# ==================== Prompt Construction Functions ====================

def construct_prompt_group_a(dialogue_history, profile_info):
    """
    Group A: Plain text + life_stage + event (force exclude NER and emotion)
    """
    prompt = f"""You are generating the interviewer's next response in a conversational interview.

Given the complete conversation history below, generate the interviewer's next question or statement as plain text.

IMPORTANT CONSTRAINTS:
- Do NOT analyze or extract named entities
- Do NOT identify or label emotions
- Focus ONLY on generating natural conversational flow based on life_stage and event context
- Output format: Plain text string only

Context:
- Session: {profile_info.get('id', 'N/A')}
- Subject: {profile_info.get('name', 'N/A')}, {profile_info.get('age', 'N/A')}, {profile_info.get('role', 'N/A')}

Complete Conversation History:
{dialogue_history}

Generate the interviewer's next response:
Output format: {{"text": "your generated response"}}
"""
    return prompt


def construct_prompt_group_b(dialogue_history, profile_info):
    """
    Group B: Text + life_stage + event + entities
    """
    prompt = f"""You are generating the interviewer's next response in a conversational interview with named entity recognition.

Given the complete conversation history and entities mentioned, generate the interviewer's next response with entity annotations.

Context:
- Session: {profile_info.get('id', 'N/A')}
- Subject: {profile_info.get('name', 'N/A')}, {profile_info.get('age', 'N/A')}, {profile_info.get('role', 'N/A')}

Complete Conversation History with Entities:
{dialogue_history}

Generate the interviewer's next response:
Output format:
{{
  "text": "your generated response",
  "entities": [
    {{
      "text": "entity text",
      "type": "PERSON|LOCATION|DATE|ORGANIZATION|etc",
      "role": "contextual role"
    }}
  ]
}}

Note: Extract all named entities from your generated text.
"""
    return prompt


def construct_prompt_group_c(dialogue_history, profile_info):
    """
    Group C: Text + life_stage + event + emotions
    """
    prompt = f"""You are generating the interviewer's next response in a conversational interview with emotion analysis.

Given the complete conversation history and emotional context, generate the interviewer's next response with emotional tone annotations.

Context:
- Session: {profile_info.get('id', 'N/A')}
- Subject: {profile_info.get('name', 'N/A')}, {profile_info.get('age', 'N/A')}, {profile_info.get('role', 'N/A')}

Complete Conversation History with Emotions:
{dialogue_history}

Generate the interviewer's next response:
Output format:
{{
  "text": "your generated response",
  "emotions": ["emotion1", "emotion2"]
}}

Note: Identify the emotional tone/intent of your generated interviewer response.
"""
    return prompt


def construct_prompt_group_d(dialogue_history, profile_info):
    """
    Group D: Text + life_stage + event + entities + emotions (full version)
    """
    prompt = f"""You are generating the interviewer's next response in a conversational interview with full annotations.

Given the complete conversation history with entities and emotions, generate the interviewer's next response with complete annotations.

Context:
- Session: {profile_info.get('id', 'N/A')}
- Subject: {profile_info.get('name', 'N/A')}, {profile_info.get('age', 'N/A')}, {profile_info.get('role', 'N/A')}

Complete Conversation History:
{dialogue_history}

Generate the interviewer's next response:
Output format:
{{
  "text": "your generated response",
  "annotations": {{
    "emotions": ["emotion1", "emotion2"],
    "entities": [
      {{
        "text": "entity text",
        "type": "PERSON|LOCATION|DATE|ORGANIZATION|etc",
        "role": "contextual role"
      }}
    ]
  }}
}}
"""
    return prompt


# ==================== Main Execution Logic ====================

def run_experiment():
    """
    Execute main workflow: Load metadata, construct ABCD prompts for all test samples.
    """
    print(f"Starting experiment prompt generation...")

    # 1. Check metadata file
    if not os.path.exists(METADATA_FILE):
        print(f"Fatal error: Metadata file '{METADATA_FILE}' not found.")
        print(f"   Please run data.py first to generate metadata.")
        return

    # 2. Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    experiment_samples = []
    
    # 3. Iterate through all files and prediction points
    for filename, meta_data in metadata.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        # Load complete dialogue data
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                full_script = json.load(f)
                all_turns = full_script.get("dialogue_turns", [])
                profile_info = full_script.get("profile", {})
        except Exception as e:
            print(f"Warning: Unable to load file {filename}. Skipping. Error: {e}")
            continue

        # 4. Iterate through all prediction points in this file
        for subject_turn_id_str, target_interviewer_id in meta_data["prediction_points"].items():
            subject_turn_id = int(subject_turn_id_str)
            
            # Get target question (Ground Truth)
            target_turn = next((t for t in all_turns if t["turn_id"] == target_interviewer_id), None)
            target_question = target_turn.get("text", "N/A") if target_turn else "N/A"

            # 5. Generate different prompts for ABCD groups
            sample = {
                "file_id": filename,
                "subject_turn": subject_turn_id,
                "target_turn": target_interviewer_id,
                "target_question_gt": target_question,
                "prompts": {}
            }
            
            # Group A: Plain text + basic annotations (force exclude entities and emotions)
            history_a = get_complete_dialogue_history(all_turns, subject_turn_id, 'A')
            sample["prompts"]["A"] = construct_prompt_group_a(history_a, profile_info)
            
            # Group B: Text + entities
            history_b = get_complete_dialogue_history(all_turns, subject_turn_id, 'B')
            sample["prompts"]["B"] = construct_prompt_group_b(history_b, profile_info)
            
            # Group C: Text + emotions
            history_c = get_complete_dialogue_history(all_turns, subject_turn_id, 'C')
            sample["prompts"]["C"] = construct_prompt_group_c(history_c, profile_info)
            
            # Group D: Complete annotations
            history_d = get_complete_dialogue_history(all_turns, subject_turn_id, 'D')
            sample["prompts"]["D"] = construct_prompt_group_d(history_d, profile_info)
            
            experiment_samples.append(sample)

    # 6. Save experiment input file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_samples, f, indent=2, ensure_ascii=False)
        
    print(f"\nComplete! Successfully generated ABCD prompts for {len(experiment_samples)} samples.")
    print(f"Experiment input file saved as '{OUTPUT_FILE}'.")
    print(f"\nStatistics:")
    print(f"   - Number of test files: {len(metadata)}")
    print(f"   - Total prediction points: {len(experiment_samples)}")
    print(f"   - Total prompts: {len(experiment_samples) * 4} (4 groups per prediction point)")


if __name__ == "__main__":
    run_experiment()
