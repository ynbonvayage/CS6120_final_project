import os
import json
import random
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = ""  # 替换您的 Key
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o"
OUTPUT_DIR = "unique_50_roles_dataset"

# ================= 1. 50 个完全不同的职业人设库 =================
# 涵盖：蓝领、白领、艺术、医疗、公共服务、科学、农业等所有领域
SCENARIOS_50 = [
    # --- Group 1: The Originals (1-10) ---
    {"id": "01_Elena", "name": "Elena", "age": 78, "role": "Retired Seamstress", "timeline": [{"stage": "Youth", "topic": "Immigrating in 1965", "context": "Alone and scared."}, {"stage": "Adulthood", "topic": "Becoming Union Rep", "context": "Fighting for fair pay."}, {"stage": "Old Age", "topic": "Teaching sewing", "context": "Passing on skills."}]},
    {"id": "02_Robert", "name": "Robert", "age": 85, "role": "Former Coal Miner", "timeline": [{"stage": "Youth", "topic": "First day underground", "context": "Fear of the dark."}, {"stage": "Adulthood", "topic": "The 1974 Strike", "context": "Violence and brotherhood."}, {"stage": "Old Age", "topic": "Town decline", "context": "Loss of community."}]},
    {"id": "03_Anita", "name": "Dr. Anita", "age": 82, "role": "Retired Biologist", "timeline": [{"stage": "Youth", "topic": "Being the only woman in lab", "context": "Sexism in the 60s."}, {"stage": "Adulthood", "topic": "Her big discovery", "context": "Professional triumph."}, {"stage": "Old Age", "topic": "Mentoring women", "context": "Legacy."}]},
    {"id": "04_Carlos", "name": "Carlos", "age": 74, "role": "Jazz Musician", "timeline": [{"stage": "Youth", "topic": "Touring in a van", "context": "Hunger and passion."}, {"stage": "Adulthood", "topic": "Hearing loss scare", "context": "Career crisis."}, {"stage": "Old Age", "topic": "Playing for himself", "context": "Music as therapy."}]},
    {"id": "05_Sarah", "name": "Sarah", "age": 74, "role": "Nurse", "timeline": [{"stage": "Adulthood", "topic": "Divorce at 40", "context": "Starting over."}, {"stage": "Adulthood", "topic": "The AIDS crisis", "context": "Compassion in tragedy."}, {"stage": "Old Age", "topic": "Retirement travel", "context": "Freedom."}]},
    {"id": "06_Kenji", "name": "Kenji", "age": 70, "role": "Tech Engineer", "timeline": [{"stage": "Youth", "topic": "Silicon Valley garage days", "context": "Innovation boom."}, {"stage": "Adulthood", "topic": "Dot-com crash", "context": "Losing fortune."}, {"stage": "Old Age", "topic": "Gardening", "context": "Digital detox."}]},
    {"id": "07_Martha", "name": "Martha", "age": 88, "role": "Farm Owner", "timeline": [{"stage": "Childhood", "topic": "No electricity", "context": "Hard simple life."}, {"stage": "Adulthood", "topic": "Saving the harvest", "context": "Fighting a drought."}, {"stage": "Old Age", "topic": "Selling the land", "context": "Grief and acceptance."}]},
    {"id": "08_Thomas", "name": "Thomas", "age": 72, "role": "Vietnam Veteran", "timeline": [{"stage": "Youth", "topic": "Returning home 1971", "context": "Alienation."}, {"stage": "Adulthood", "topic": "Building his house", "context": "Healing through labor."}, {"stage": "Old Age", "topic": "Platoon reunion", "context": "Forgiveness."}]},
    {"id": "09_Wei", "name": "Wei", "age": 82, "role": "Restaurant Owner", "timeline": [{"stage": "Adulthood", "topic": "Opening day 1975", "context": "High risk."}, {"stage": "Adulthood", "topic": "Kitchen fire", "context": "Rebuilding."}, {"stage": "Old Age", "topic": "Closing shop", "context": "End of an era."}]},
    {"id": "10_Lydia", "name": "Lydia", "age": 76, "role": "Retired Teacher", "timeline": [{"stage": "Youth", "topic": "Peace Corps", "context": "Idealism abroad."}, {"stage": "Adulthood", "topic": "Teaching during protests", "context": "Civil rights era."}, {"stage": "Old Age", "topic": "Writing memoirs", "context": "Reflection."}]},
    
    # --- Group 2: Public Service & Trades (11-20) ---
    {"id": "11_Frank", "name": "Frank", "age": 75, "role": "Retired Firefighter", "timeline": [{"stage": "Youth", "topic": "Rookie year training", "context": "Physical exhaustion."}, {"stage": "Adulthood", "topic": "The Big Fire of '88", "context": "Saving a family."}, {"stage": "Old Age", "topic": "Health issues", "context": "Lungs and legacy."}]},
    {"id": "12_Alice", "name": "Alice", "age": 79, "role": "Journalist", "timeline": [{"stage": "Youth", "topic": "Copy girl in newsroom", "context": "Fighting to write."}, {"stage": "Adulthood", "topic": "Covering Watergate era", "context": "Pursuit of truth."}, {"stage": "Old Age", "topic": "The death of print", "context": "Changing media."}]},
    {"id": "13_Hiro", "name": "Hiro", "age": 81, "role": "Architect", "timeline": [{"stage": "Youth", "topic": "Studying in Europe", "context": "Inspiration."}, {"stage": "Adulthood", "topic": "Designing the City Library", "context": "Career peak."}, {"stage": "Old Age", "topic": "Seeing his buildings age", "context": "Legacy vs decay."}]},
    {"id": "14_Joe", "name": "Joe", "age": 73, "role": "Truck Driver", "timeline": [{"stage": "Youth", "topic": "First cross-country trip", "context": "Freedom of the road."}, {"stage": "Adulthood", "topic": "Missing kids' birthdays", "context": "Sacrifice for money."}, {"stage": "Old Age", "topic": "Losing his license", "context": "Loss of independence."}]},
    {"id": "15_Mary", "name": "Mary", "age": 84, "role": "Librarian", "timeline": [{"stage": "Youth", "topic": "The quiet reading room", "context": "Love for books."}, {"stage": "Adulthood", "topic": "The computer revolution", "context": "Adapting to digital."}, {"stage": "Old Age", "topic": "The library closing", "context": "Community loss."}]},
    {"id": "16_Pierre", "name": "Pierre", "age": 77, "role": "Baker", "timeline": [{"stage": "Youth", "topic": "Apprenticeship in Paris", "context": "Strict discipline."}, {"stage": "Adulthood", "topic": "Opening NY bakery", "context": "Culture shock."}, {"stage": "Old Age", "topic": "Losing sense of smell", "context": "Adapting craft."}]},
    {"id": "17_Mike", "name": "Mike", "age": 71, "role": "Police Officer", "timeline": [{"stage": "Youth", "topic": "Beat cop in the 70s", "context": "Dangerous streets."}, {"stage": "Adulthood", "topic": "Making Detective", "context": "The case that haunts him."}, {"stage": "Old Age", "topic": "Retirement fishing", "context": "Finding peace."}]},
    {"id": "18_Stan", "name": "Stan", "age": 80, "role": "Steel Worker", "timeline": [{"stage": "Youth", "topic": "Following dad to the mill", "context": "Tradition."}, {"stage": "Adulthood", "topic": "The Rust Belt crash", "context": "Unemployment."}, {"stage": "Old Age", "topic": "Health checkups", "context": "Physical toll."}]},
    {"id": "19_Betty", "name": "Betty", "age": 83, "role": "Flight Attendant", "timeline": [{"stage": "Youth", "topic": "Golden Age of Flying", "context": "Glamour and travel."}, {"stage": "Adulthood", "topic": "Deregulation era", "context": "Strikes and chaos."}, {"stage": "Old Age", "topic": "Grounding", "context": "Missing the sky."}]},
    {"id": "20_Raj", "name": "Raj", "age": 78, "role": "Surgeon", "timeline": [{"stage": "Youth", "topic": "Medical school stress", "context": "No sleep."}, {"stage": "Adulthood", "topic": "First heart transplant", "context": "Life and death."}, {"stage": "Old Age", "topic": "Shaking hands", "context": "Retiring the scalpel."}]},

    # --- Group 3: Specialized & Arts (21-30) ---
    {"id": "21_Lars", "name": "Lars", "age": 75, "role": "Fisherman", "timeline": [{"stage": "Youth", "topic": "The Great Storm", "context": "Survival at sea."}, {"stage": "Adulthood", "topic": "Buying his own boat", "context": "Independence."}, {"stage": "Old Age", "topic": "Overfishing regulations", "context": "Changing industry."}]},
    {"id": "22_Jack", "name": "Jack", "age": 86, "role": "Carpenter", "timeline": [{"stage": "Youth", "topic": "Building his own house", "context": "Self-reliance."}, {"stage": "Adulthood", "topic": "Injury on the job", "context": "Financial fear."}, {"stage": "Old Age", "topic": "Carving toys", "context": "Gentle work."}]},
    {"id": "23_Eleanor", "name": "Eleanor", "age": 88, "role": "Diplomat", "timeline": [{"stage": "Youth", "topic": "Cold War posting", "context": "Espionage fears."}, {"stage": "Adulthood", "topic": "Peace treaty negotiation", "context": "High stakes."}, {"stage": "Old Age", "topic": "Writing memoirs", "context": "Secrets kept."}]},
    {"id": "24_Yuri", "name": "Yuri", "age": 72, "role": "Ballet Dancer", "timeline": [{"stage": "Youth", "topic": "Defecting to the West", "context": "Leaving family."}, {"stage": "Adulthood", "topic": "The knee injury", "context": "Career ending."}, {"stage": "Old Age", "topic": "Teaching kids", "context": "Finding joy again."}]},
    {"id": "25_Ahmed", "name": "Ahmed", "age": 76, "role": "Taxi Driver", "timeline": [{"stage": "Youth", "topic": "Driving in 80s NYC", "context": "Chaos and crime."}, {"stage": "Adulthood", "topic": "Saving for a medallion", "context": "The American Dream."}, {"stage": "Old Age", "topic": "Uber taking over", "context": "Obsolescence."}]},
    {"id": "26_Sam", "name": "Sam", "age": 79, "role": "Electrician", "timeline": [{"stage": "Youth", "topic": "The Blackout of '77", "context": "Restoring power."}, {"stage": "Adulthood", "topic": "Starting own business", "context": "Stress of payroll."}, {"stage": "Old Age", "topic": "Wiring his son's house", "context": "Fatherly pride."}]},
    {"id": "27_Cliff", "name": "Cliff", "age": 84, "role": "Postman", "timeline": [{"stage": "Youth", "topic": "Walking the route", "context": "Knowing everyone."}, {"stage": "Adulthood", "topic": "The dog bite incident", "context": "Hazard of the job."}, {"stage": "Old Age", "topic": "Letters vs Email", "context": "Nostalgia."}]},
    {"id": "28_Donna", "name": "Donna", "age": 73, "role": "Social Worker", "timeline": [{"stage": "Youth", "topic": "First foster case", "context": "Heartbreak."}, {"stage": "Adulthood", "topic": "System burnout", "context": "Fighting bureaucracy."}, {"stage": "Old Age", "topic": "Adopting a child", "context": "Personal redemption."}]},
    {"id": "29_Hans", "name": "Hans", "age": 89, "role": "Clockmaker", "timeline": [{"stage": "Youth", "topic": "Learning the gears", "context": "Patience."}, {"stage": "Adulthood", "topic": "The Quartz Crisis", "context": "Technology threat."}, {"stage": "Old Age", "topic": "Repairing antiques", "context": "Preserving time."}]},
    {"id": "30_Rose", "name": "Rose", "age": 81, "role": "Botanist", "timeline": [{"stage": "Youth", "topic": "Amazon expedition", "context": "Discovery."}, {"stage": "Adulthood", "topic": "Saving a species", "context": "Conservation."}, {"stage": "Old Age", "topic": "Her home garden", "context": "Personal oasis."}]},

    # --- Group 4: Service & Unique Roles (31-40) ---
    {"id": "31_Maria", "name": "Maria", "age": 75, "role": "Housekeeper", "timeline": [{"stage": "Youth", "topic": "Raising other's kids", "context": "Sacrifice."}, {"stage": "Adulthood", "topic": "Buying her own home", "context": "Achievement."}, {"stage": "Old Age", "topic": "Being visited by them", "context": "Gratitude."}]},
    {"id": "32_Arthur", "name": "Arthur", "age": 77, "role": "Banker", "timeline": [{"stage": "Youth", "topic": "Wall Street 80s", "context": "Greed and excess."}, {"stage": "Adulthood", "topic": "The 2008 Crash", "context": "Guilt and panic."}, {"stage": "Old Age", "topic": "Charity work", "context": "Atonement."}]},
    {"id": "33_Reynolds", "name": "Capt. Reynolds", "age": 80, "role": "Pilot", "timeline": [{"stage": "Youth", "topic": "First solo flight", "context": "Freedom."}, {"stage": "Adulthood", "topic": "Emergency landing", "context": "Focus under pressure."}, {"stage": "Old Age", "topic": "Failing the eye exam", "context": "Grounded."}]},
    {"id": "34_Mario", "name": "Mario", "age": 74, "role": "Plumber", "timeline": [{"stage": "Youth", "topic": "Helping his father", "context": "Dirty work."}, {"stage": "Adulthood", "topic": "The great flood of '96", "context": "Saving the basement."}, {"stage": "Old Age", "topic": "Knees giving out", "context": "Physical cost."}]},
    {"id": "35_Gina", "name": "Gina", "age": 70, "role": "Hairdresser", "timeline": [{"stage": "Youth", "topic": "Beauty school", "context": "Dreams of glamour."}, {"stage": "Adulthood", "topic": "Secrets in the chair", "context": "Being a therapist."}, {"stage": "Old Age", "topic": "Closing the salon", "context": "Missing the gossip."}]},
    {"id": "36_Dave", "name": "Dave", "age": 72, "role": "Welder", "timeline": [{"stage": "Youth", "topic": "Building skyscrapers", "context": "No fear of heights."}, {"stage": "Adulthood", "topic": "Flash burn injury", "context": "Danger of the job."}, {"stage": "Old Age", "topic": "Sculpting metal art", "context": "New expression."}]},
    {"id": "37_Saul", "name": "Saul", "age": 83, "role": "Lawyer", "timeline": [{"stage": "Youth", "topic": "Public defender days", "context": "Fighting the system."}, {"stage": "Adulthood", "topic": "The big civil rights case", "context": "Justice."}, {"stage": "Old Age", "topic": "Teaching ethics", "context": "Warning the next gen."}]},
    {"id": "38_Helen", "name": "Helen", "age": 78, "role": "Magazine Editor", "timeline": [{"stage": "Youth", "topic": "The fashion closet", "context": "Hard work, low pay."}, {"stage": "Adulthood", "topic": "Print to Digital shift", "context": "Adapt or die."}, {"stage": "Old Age", "topic": "Reading for pleasure", "context": "Rediscovering love of words."}]},
    {"id": "39_Steve", "name": "Steve", "age": 75, "role": "Zookeeper", "timeline": [{"stage": "Youth", "topic": "Night shift with lions", "context": "Respect for nature."}, {"stage": "Adulthood", "topic": "The animal escape", "context": "Crisis."}, {"stage": "Old Age", "topic": "Retirement volunteering", "context": "Can't stay away."}]},
    {"id": "40_Indira", "name": "Indira", "age": 79, "role": "Archaeologist", "timeline": [{"stage": "Youth", "topic": "The first dig in Egypt", "context": "Wonder."}, {"stage": "Adulthood", "topic": "Fighting looters", "context": "Protecting history."}, {"stage": "Old Age", "topic": "Museum curator", "context": "Preservation."}]},
    
    # --- Group 5: Miscellaneous (41-50) ---
    {"id": "41_Mo", "name": "Mo", "age": 71, "role": "Bartender", "timeline": [{"stage": "Youth", "topic": "The disco era", "context": "Wild nights."}, {"stage": "Adulthood", "topic": "Buying the bar", "context": "Responsibility."}, {"stage": "Old Age", "topic": "Sober now", "context": "Irony."}]},
    {"id": "42_Ruth", "name": "Ruth", "age": 85, "role": "Judge", "timeline": [{"stage": "Youth", "topic": "Law school discrimination", "context": "Being the only woman."}, {"stage": "Adulthood", "topic": "The death penalty case", "context": "Heavy burden."}, {"stage": "Old Age", "topic": "Retirement hobbies", "context": "Gardening judgment free."}]},
    {"id": "43_Al", "name": "Al", "age": 76, "role": "Auto Mechanic", "timeline": [{"stage": "Youth", "topic": "Muscle car era", "context": "Speed and grease."}, {"stage": "Adulthood", "topic": "Computerized engines", "context": "Learning curve."}, {"stage": "Old Age", "topic": "Restoring a classic", "context": "Passion project."}]},
    {"id": "44_Sophie", "name": "Sophie", "age": 73, "role": "Hotel Manager", "timeline": [{"stage": "Youth", "topic": "Maid service", "context": "Invisible work."}, {"stage": "Adulthood", "topic": "The celebrity scandal", "context": "Discretion."}, {"stage": "Old Age", "topic": "Bed and Breakfast owner", "context": "Personal touch."}]},
    {"id": "45_George", "name": "George", "age": 80, "role": "Train Conductor", "timeline": [{"stage": "Youth", "topic": "Night trains", "context": "Romance of rails."}, {"stage": "Adulthood", "topic": "The derailment", "context": "Trauma and heroism."}, {"stage": "Old Age", "topic": "Model trains", "context": "Small scale control."}]},
    {"id": "46_Fiona", "name": "Fiona", "age": 74, "role": "Potter", "timeline": [{"stage": "Youth", "topic": "Living in a commune", "context": "Art over money."}, {"stage": "Adulthood", "topic": "The gallery fire", "context": "Losing work."}, {"stage": "Old Age", "topic": "Arthritis in hands", "context": "Adapting technique."}]},
    {"id": "47_Tom", "name": "Tom", "age": 77, "role": "Coast Guard", "timeline": [{"stage": "Youth", "topic": "Storm rescue", "context": "Adrenaline."}, {"stage": "Adulthood", "topic": "Oil spill cleanup", "context": "Environmental grief."}, {"stage": "Old Age", "topic": "Living by the lake", "context": "Respect for water."}]},
    {"id": "48_Nora", "name": "Nora", "age": 82, "role": "Midwife", "timeline": [{"stage": "Youth", "topic": "First home birth", "context": "Miracle of life."}, {"stage": "Adulthood", "topic": "Complications", "context": "Tragedy and resilience."}, {"stage": "Old Age", "topic": "Meeting babies grown up", "context": "Cycle of life."}]},
    {"id": "49_Lin", "name": "Lin", "age": 75, "role": "Translator", "timeline": [{"stage": "Youth", "topic": "Immigrating alone", "context": "Language as a bridge."}, {"stage": "Adulthood", "topic": "High stakes UN meeting", "context": "Pressure."}, {"stage": "Old Age", "topic": "Forgetting native words", "context": "Fear of aging."}]},
    {"id": "50_Butch", "name": "Butch", "age": 79, "role": "Butcher", "timeline": [{"stage": "Youth", "topic": "Family shop apprentice", "context": "Tradition."}, {"stage": "Adulthood", "topic": "Supermarkets taking over", "context": "Competition."}, {"stage": "Old Age", "topic": "Vegan granddaughter", "context": "Cultural shift."}]}
]

TONE_VARIATIONS = [
    "Talkative and Detailed (Eager to share every memory)",      # 健谈细节控：回答很长，实体很多
    "Short-spoken and Reserved (Needs probing)",                 # 寡言保留：回答简短，需要主持人追问 (测试 QG 能力)
    "Warm and Grandmotherly/Grandfatherly (Soft spoken)",        # 慈祥温暖：语气柔和，充满关怀
    "Frank and Direct (No sugar-coating)",                       # 直率坦诚：不加修饰，有什么说什么
    "Slightly Wandering and Forgetful (Goes off-topic)",         # 容易跑题/健忘：模拟真实老人的思维跳跃
    "Emotional and Vulnerable (Open about feelings)",            # 易感脆弱：容易流露深层情感
    "Humorous and Self-deprecating (Jokes about hardships)",     # 幽默自嘲：用笑声掩盖苦难 (测试情感分析的各种 Nuance)
    "Proud and Dignified (Focuses on achievements)",             # 骄傲庄重：强调成就和尊严
    "Melancholic and Slow-paced (Deep in thought)",              # 忧郁缓慢：思考时间长，语调低沉
    "Matter-of-fact (Treats the interview like a report)"        # 公事公办：像做报告一样陈述事实
]

# ================= 2. 动态节奏控制 (10-15 turns) =================
def get_dynamic_instruction(current_idx, total_turns, timeline):
    available_slots = total_turns - 1
    base_len = available_slots // 3
    remainder = available_slots % 3
    
    len_s1 = base_len + (1 if remainder > 0 else 0)
    len_s2 = base_len + (1 if remainder > 1 else 0)
    
    end_s1 = len_s1
    end_s2 = len_s1 + len_s2

    s1, s2, s3 = timeline[0], timeline[1], timeline[2]

    if current_idx == total_turns - 1:
        return "FINAL PHASE: SUMMARY. Use 'Before we finish...' to ask for a final lesson.", "Old Age"

    if current_idx < end_s1:
        label = s1['stage']
        if current_idx == 0: instr = f"PHASE 1 (START): Use 'I'd love to hear about...' to start topic: {s1['topic']}."
        elif current_idx == end_s1 - 1: instr = f"PHASE 1 (CLOSE): Ask how this period ended."
        else: instr = f"PHASE 1 (DEEPEN): Use 'It sounds like...' to probe."
            
    elif current_idx < end_s2:
        label = s2['stage']
        if current_idx == end_s1: instr = f"PHASE 2 (TRANSITION): Use 'So when did things change?' to move to: {s2['topic']}."
        else: instr = f"PHASE 2 (EVENT): Discuss core challenges."

    else:
        label = s3['stage']
        if current_idx == end_s2: instr = f"PHASE 3 (TRANSITION): Move to present day: {s3['topic']}."
        else: instr = f"PHASE 3 (REFLECTION): Discuss feelings."

    return instr, label

# ================= 3. 生成主函数 =================
def generate_session(scenario, tone):
    total_q_pairs = random.randint(10, 15)
    print(f"🎲 Generating {scenario['id']} ({tone}) | Length: {total_q_pairs}")
    
    conversation_history = []
    dialogue_turns = []
    
    for i in range(total_q_pairs): 
        instruction, stage_label = get_dynamic_instruction(i, total_q_pairs, scenario['timeline'])
        
        system_prompt = f"""
        You are simulating a professional Oral History Interview.
        
        ROLE 1: INTERVIEWER (Archivist)
        - Persona: Professional, neutral, empathetic but RESTRAINED.
        - Style: Natural responses ("That must have been hard"). Colloquial.
        - Transitions: "I'd love to hear...", "So when did things change?".
        - Constraint: NO Future Knowledge.
        
        ROLE 2: SUBJECT ({scenario['name']}, {scenario['age']}, {scenario['role']})
        - Context Timeline: {scenario['timeline'][0]['topic']} -> {scenario['timeline'][1]['topic']} -> {scenario['timeline'][2]['topic']}
        - Current Instruction: {instruction}
        - **Personality Tone**: {tone}
        - **Emotion Rule**: Natural, authentic. Do not force drama.
        
        OUTPUT JSON:
        {{
            "interviewer_text": "...",
            "subject_text": "...",
            "subject_annotations": [
                {{
                    "text": "...",
                    "life_stage": "{stage_label}", 
                    "event_type": "...",
                    "emotions": ["..."], 
                    "entities": [...]
                }}
            ]
        }}
        """
        
        recent_history = conversation_history[-10:]
        history_text = "\n".join(recent_history)
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"History Snippet:\n{history_text}\n\nGenerate Pair {i+1}/{total_q_pairs}:"}
                ],
                response_format={"type": "json_object"},
                temperature=0.75
            )
            data = json.loads(response.choices[0].message.content)
            
            dialogue_turns.append({"turn_id": i*2+1, "speaker": "Interviewer", "text": data['interviewer_text']})
            dialogue_turns.append({"turn_id": i*2+2, "speaker": "Subject", "text": data['subject_text'], "sentence_annotations": data['subject_annotations']})
            conversation_history.append(f"I: {data['interviewer_text']}\nS: {data['subject_text']}")
            
        except Exception as e:
            print(f"Error: {e}")
            break
            
    return {"session_id": scenario['id'], "tone": tone, "profile": scenario, "dialogue_turns": dialogue_turns}

# ================= 4. 执行 =================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"🚀 Processing 50 Unique Scenarios...")

    for scenario in SCENARIOS_50:
        tone = random.choice(TONE_VARIATIONS)
        res = generate_session(scenario, tone)
        
        if res:
            filename = f"{OUTPUT_DIR}/{scenario['id']}_{tone.split()[0]}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {filename}")