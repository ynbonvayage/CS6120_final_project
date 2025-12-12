import os
import json

json_folder = 'processed_emotion/'

output_folder = 'final_processed_emotion/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(json_folder):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_folder, file_name)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for turn in data['dialogue_turns']:
            if turn['speaker'] == 'Subject':
                if 'sentences' in turn:
                    for sentence in turn['sentences']:
                        sentence['annotations']['emotions'] = sentence['predicted_emotions']
                        del sentence['predicted_emotions']
        
        output_file_path = os.path.join(output_folder, file_name)
        with open(output_file_path, 'w') as file:  
            json.dump(data, file, indent=2)

print("All JSON files processed successfully and saved in the 'final_processed_emotion' folder.")