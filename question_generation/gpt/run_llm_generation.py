import json
import os
from openai import OpenAI
import time
from datetime import datetime

# ========================= Configuration Loading =========================
# Try to import configuration from config.py, use default values if not found
try:
    from config import (
        OPENAI_API_KEY,
        LLM_MODEL,
        TEMPERATURE,
        MAX_RETRIES,
        RETRY_DELAY,
        REQUEST_DELAY,
        INPUT_FILE,
        OUTPUT_FILE,
        CHECKPOINT_FILE
    )
    print("Loaded configuration from config.py")
except ImportError:
    print("config.py not found, using default configuration")
    print("   Tip: Copy config_template.py to config.py and fill in your configuration")
    
    # Default configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # put api key here
    LLM_MODEL = "gpt-4o"
    TEMPERATURE = 0.7
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    REQUEST_DELAY = 1
    INPUT_FILE = "../experiment_3_prompts_input.json"
    OUTPUT_FILE = "experiment_3_generation_results.json"
    CHECKPOINT_FILE = "experiment_3_checkpoint.json"

# Validate API Key
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
    print("\nError: No valid OPENAI_API_KEY configured")
    print("   Please set your API Key in config.py, or set environment variable OPENAI_API_KEY")
    exit(1)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"\nError: Unable to initialize OpenAI client: {e}")
    exit(1)

# ==================== LLM Call Functions ====================

def generate_question_from_prompt(prompt, retry_count=0):
    """
    Call LLM API to generate questions based on the prompt.
    Supports automatic retry mechanism.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional oral history interviewer. Generate natural, thoughtful follow-up questions based on the conversation history provided."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON format output (if any)
        try:
            # Check if it's JSON format
            if generated_text.startswith('{') and generated_text.endswith('}'):
                parsed_json = json.loads(generated_text)
                return {
                    "raw_output": generated_text,
                    "parsed": parsed_json,
                    "format": "json"
                }
        except json.JSONDecodeError:
            pass
        
        # Return plain text format
        return {
            "raw_output": generated_text,
            "parsed": None,
            "format": "text"
        }
        
    except Exception as e:
        error_msg = f"API_ERROR: {str(e)}"
        print(f"      Error: {error_msg}")
        
        # If retry attempts remain, wait and retry
        if retry_count < MAX_RETRIES:
            print(f"      Retrying in {RETRY_DELAY} seconds ({retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return generate_question_from_prompt(prompt, retry_count + 1)
        
        # Retry attempts exhausted, return error message
        return {
            "raw_output": error_msg,
            "parsed": None,
            "format": "error"
        }


def save_checkpoint(results, checkpoint_file):
    """Save checkpoint data for resuming after interruption"""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_file):
    """Load checkpoint data"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ==================== Main Execution Logic ====================

def run_generation():
    """
    Execute Step 3: Iterate through all prompts and call LLM API to generate questions.
    """
    print("=" * 80)
    print("Step 3: LLM Question Generation Started")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using model: {LLM_MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    
    # 1. Check input file
    if not os.path.exists(INPUT_FILE):
        print(f"\nFatal error: Input file '{INPUT_FILE}' not found.")
        print(f"   Please run run_experiment.py first to generate the input file.")
        return

    # 2. Load input data
    print(f"\nLoading input file: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    total_samples = len(samples)
    total_calls = total_samples * 4  # Groups A, B, C, D
    
    print(f"Statistics:")
    print(f"   - Number of samples: {total_samples}")
    print(f"   - API calls: {total_calls} (4 groups per sample)")
    print(f"   - Estimated time: ~{total_calls * REQUEST_DELAY / 60:.1f} minutes")
    
    # 3. Check for checkpoint file (support resume)
    checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint_data:
        response = input(f"\nCheckpoint file found. Continue from last progress? (y/n): ")
        if response.lower() == 'y':
            results = checkpoint_data
            print(f"Loaded {len(results)} completed samples")
        else:
            results = []
            print("Starting from beginning...")
    else:
        results = []
    
    start_idx = len(results)
    start_time = time.time()
    
    # 4. Iterate through all samples
    for i in range(start_idx, total_samples):
        sample = samples[i]
        
        print("\n" + "=" * 80)
        print(f"Processing sample {i+1}/{total_samples}")
        print(f"   File: {sample['file_id']}")
        print(f"   Subject Turn: {sample['subject_turn']} -> Target Turn: {sample['target_turn']}")
        print(f"   Ground Truth: {sample['target_question_gt'][:80]}...")
        print("-" * 80)
        
        run_data = {
            "sample_id": i + 1,
            "file_id": sample["file_id"],
            "subject_turn": sample["subject_turn"],
            "target_turn": sample["target_turn"],
            "target_question_gt": sample["target_question_gt"],
            "generated_questions": {},
            "generation_timestamp": datetime.now().isoformat()
        }
        
        # Iterate through groups A, B, C, D
        for group in ['A', 'B', 'C', 'D']:
            prompt = sample["prompts"][group]
            
            print(f"\n   Generating group {group}...", end=" ")
            
            # Call LLM API
            result = generate_question_from_prompt(prompt)
            
            # Save generation result
            run_data["generated_questions"][group] = result
            
            # Display generated question (truncated)
            if result["format"] == "error":
                print(f"Failed")
                print(f"      Error: {result['raw_output'][:60]}...")
            else:
                generated_text = result["raw_output"]
                print(f"Success")
                print(f"      Generated: {generated_text[:80]}...")
            
            # Avoid API rate limit
            if group != 'D':  # No need to wait after last request
                time.sleep(REQUEST_DELAY)
        
        results.append(run_data)
        
        # Save checkpoint after each sample
        save_checkpoint(results, CHECKPOINT_FILE)
        
        # Display progress
        elapsed = time.time() - start_time
        avg_time_per_sample = elapsed / (i - start_idx + 1)
        remaining_samples = total_samples - (i + 1)
        eta_seconds = avg_time_per_sample * remaining_samples
        
        print(f"\n   Progress: {i+1}/{total_samples} ({(i+1)/total_samples*100:.1f}%)")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Remaining: {eta_seconds/60:.1f} minutes")

    end_time = time.time()
    total_time = end_time - start_time
    
    # 5. Save final results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 6. Delete checkpoint file (completed)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    # 7. Generate statistics report
    print("\n" + "=" * 80)
    print("Step 3 Complete!")
    print("=" * 80)
    
    # Count successful and failed API calls
    total_generated = 0
    total_errors = 0
    json_format_count = 0
    text_format_count = 0
    
    for result in results:
        for group in ['A', 'B', 'C', 'D']:
            gen = result["generated_questions"][group]
            if gen["format"] == "error":
                total_errors += 1
            else:
                total_generated += 1
                if gen["format"] == "json":
                    json_format_count += 1
                else:
                    text_format_count += 1
    
    print(f"Generation Statistics:")
    print(f"   - Total samples: {len(results)}")
    print(f"   - Successfully generated: {total_generated}/{total_calls} ({total_generated/total_calls*100:.1f}%)")
    print(f"   - Failed: {total_errors}")
    print(f"   - JSON format: {json_format_count}")
    print(f"   - Text format: {text_format_count}")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


# ==================== Entry Point ====================

if __name__ == "__main__":
    try:
        run_generation()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        print(f"Current progress saved to: {CHECKPOINT_FILE}")
        print("   You can choose to resume from last progress on next run")
    except Exception as e:
        print(f"\n\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()