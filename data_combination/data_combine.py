import os
import json

emotion_folder = 'emotion/'
entity_folder = 'entity/'
processed_emotion_folder = 'processed_emotion/'

if not os.path.exists(processed_emotion_folder):
    os.makedirs(processed_emotion_folder)

for emotion_file in os.listdir(emotion_folder):
    if emotion_file.endswith('_Annotated.json'):
        with open(os.path.join(emotion_folder, emotion_file), 'r') as f:
            emotion_data = json.load(f)
        
        entity_file = 'KB_' + emotion_file
        with open(os.path.join(entity_folder, entity_file), 'r') as f:
            entity_data = json.load(f)
        
        entity_dict = {}
        for item in entity_data['dialogue_content']:
            if item['speaker'] == 'Subject':  
                key = (item['turn_id'], item['original_text'])
                entity_dict[key] = item.get('predicted_entities', [])
        
        for turn in emotion_data['dialogue_turns']:
            if turn['speaker'] == 'Subject':  
                if 'sentences' in turn:
                    for sentence in turn['sentences']:
                        key = (turn['turn_id'], sentence['text'])
                        if key in entity_dict:
                            sentence['annotations']['entities'] = entity_dict[key]
                        else:
                            sentence['annotations']['entities'] = []
        
        with open(os.path.join(processed_emotion_folder, emotion_file), 'w') as f:
            json.dump(emotion_data, f, indent=2)

print("All files processed successfully and saved in the 'processed_emotion' folder.")