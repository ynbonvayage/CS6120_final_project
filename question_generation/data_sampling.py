import os
import json
import random

# ========================= Configuration Area (Adjust according to your file structure) =========================
# Directory where your 50 JSON files are located
DATA_DIR = "../data_json"
# Determine test set size: extract 20% of total
TEST_SET_PERCENTAGE = 0.20
# Number of prediction nodes to determine per interview
PREDICT_NODES_PER_FILE = 4
# Set random seed to ensure reproducibility
random.seed(42) 

# ======================= Function Definitions =======================

def get_json_files(data_dir):
    """
    Actually read all .json filenames in the data_json directory.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist. Please ensure your JSON files are uploaded to the 'data_json' folder.")
        return []
        
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    return all_files

def load_and_count_turns(filepath):
    """
    Load a single JSON file and return its total number of dialogue turns.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Total turns is the length of the dialogue_turns list
            return len(data.get("dialogue_turns", []))
    except Exception as e:
        print(f"Failed to read or parse file {filepath}: {e}")
        return 0

def create_test_set_metadata(all_files, data_dir, percentage, num_nodes_per_file):
    """
    Randomly extract test set files and determine prediction nodes for each file.
    """
    if not all_files:
        return {}

    # 1. Extract test set files
    test_set_size = int(len(all_files) * percentage)
    test_files = random.sample(all_files, test_set_size)

    test_set_metadata = {}

    for filename in test_files:
        filepath = os.path.join(data_dir, filename)
        total_turns = load_and_count_turns(filepath)
        
        if total_turns == 0:
            print(f"Skipping file {filename} due to zero turn count.")
            continue

        # Find all "Subject speech" turn IDs (Subject Turns)
        # Subject Turns are all even IDs (2, 4, 6, ..., total_turns).
        # Exclude last few turns to ensure sufficient space for prediction and history review (e.g., at least need Turn 4 history)
        subject_turns = [i for i in range(2, total_turns) if i % 2 == 0]
        
        # Ensure prediction points are at least after Turn 4 to have sufficient historical context
        available_nodes = [t_id for t_id in subject_turns if t_id >= 4]

        # Randomly select prediction nodes
        if len(available_nodes) < num_nodes_per_file:
            selected_nodes = available_nodes
        else:
            selected_nodes = random.sample(available_nodes, num_nodes_per_file)
            
        # Record prediction points as: prediction point (Subject Turn ID) -> target question (Interviewer Turn ID)
        prediction_points = {
            t_id: t_id + 1 for t_id in sorted(selected_nodes)
        }

        test_set_metadata[filename] = {
            "total_turns": total_turns,
            "prediction_points": prediction_points
        }
        
    return test_set_metadata

# ======================= Main Execution Logic =======================

if __name__ == "__main__":
    print(f"Step 1: Test set extraction and prediction point determination started...")
    
    all_json_files = get_json_files(DATA_DIR)
    
    if not all_json_files:
        print("Terminating script: No JSON files found.")
    elif len(all_json_files) != 50:
        print(f"Warning: Found {len(all_json_files)} files in '{DATA_DIR}' directory, not the expected 50. Continuing.")
    else:
        print(f"Successfully found {len(all_json_files)} JSON files.")

    test_metadata = create_test_set_metadata(
        all_json_files, 
        DATA_DIR,
        TEST_SET_PERCENTAGE, 
        PREDICT_NODES_PER_FILE
    )
    
    # Print final metadata report
    final_report = []
    total_samples = 0
    
    for filename, data in test_metadata.items():
        points = data['prediction_points']
        total_samples += len(points)
        
        point_str = ", ".join([f"Turn {sub} -> Turn {inter}" for sub, inter in points.items()])
        
        final_report.append(f"File: {filename}")
        final_report.append(f"  Total turns: {data['total_turns']}")
        final_report.append(f"  Prediction nodes (Subject Turn ID -> Target Question ID): {point_str}")
        final_report.append("-" * 50)

    print(f"\nSuccessfully extracted {len(test_metadata)} files as test set ({TEST_SET_PERCENTAGE*100:.0f}% of total).")
    print("\n--- Test Set Metadata Report ---")
    print("\n".join(final_report))
    print(f"Total prediction samples (Total Samples): {total_samples} (each sample will run 4 groups of ablation experiments)")
    
    # Recommend saving this metadata file for use in subsequent steps
    with open("experiment_3_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(test_metadata, f, indent=2, ensure_ascii=False)
    
    print("\nMetadata saved to experiment_3_metadata.json file.")