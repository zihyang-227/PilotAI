import os
import re
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. SETTINGS & CLIENT
# ==========================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.1
CHUNK_SIZE = 10 
MAX_WORKERS = 10 

# ==========================================
# 2. ROBUST VTT PARSING & MERGING
# ==========================================

def parse_and_merge_vtt(vtt_file_path):
    with open(vtt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    raw_turns = []
    current_timestamp = None
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) -->')
    speaker_pattern = re.compile(r'^([^:\n]+):\s*(.*)')

    for line in lines:
        line = line.strip()
        time_match = time_pattern.search(line)
        if time_match:
            current_timestamp = time_match.group(1)
            continue
        speaker_match = speaker_pattern.match(line)
        if speaker_match and current_timestamp:
            raw_turns.append({
                'Time': current_timestamp,
                'Speaker': speaker_match.group(1).strip(),
                'Remark': speaker_match.group(2).strip()
            })
            current_timestamp = None

    if not raw_turns: return []

    merged_turns = []
    current_turn = raw_turns[0].copy()
    for next_turn in raw_turns[1:]:
        if next_turn['Speaker'] == current_turn['Speaker']:
            current_turn['Remark'] += " " + next_turn['Remark']
        else:
            merged_turns.append(current_turn)
            current_turn = next_turn.copy()
    merged_turns.append(current_turn)
    
    for idx, turn in enumerate(merged_turns):
        turn['Turn_Number'] = idx + 1
    return merged_turns

def get_coding_scheme(prompt_csv_path):
    df = pd.read_csv(prompt_csv_path)
    scheme = []
    instructions = "Coding Instructions for EACH TURN:\n"
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        if label not in ["Unnamed", "Turn (timestamp) ", "Turn Number"]:
            prompt_text = str(row.iloc[1])
            scale = str(row.iloc[2])
            scheme.append({'label': label, 'prompt': prompt_text, 'scale': scale})
            instructions += f"- {label}: {prompt_text} (Scale: {scale})\n"
    return scheme, instructions

# ==========================================
# 3. PARALLEL WORKER FUNCTION
# ==========================================

def process_chunk(chunk_index, chunk, instruction_text, coding_scheme):
    transcript_block = ""
    for idx, t in enumerate(chunk):
        transcript_block += f"Turn {idx+1} | {t['Speaker']}: {t['Remark']}\n"

    system_prompt = (
        "You are a behavioral research assistant. Analyze the transcript chunk turn-by-turn. "
        "For the 'Episode_Shift' field, return 'NEW' if a turn marks a macroscopic change in purpose, otherwise 'SAME'. "
        "For episode-level prompts (like Resolution or Action Plan), evaluate if the current turn/context provides that info. "
        "Return a JSON object with a 'results' key containing an array of objects."
    )
    
    user_prompt = (
        f"CODING SCHEME:\n{instruction_text}\n"
        "- Episode_Shift: Mark 'NEW' for macroscopic shifts.\n\n"
        f"TRANSCRIPT CHUNK:\n{transcript_block}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            batch_results = json.loads(response.choices[0].message.content).get('results', [])
            
            processed_rows = []
            for idx, t in enumerate(chunk):
                ai_data = batch_results[idx] if idx < len(batch_results) else {}
                row = {
                    'Time': t['Time'], 'Speaker': t['Speaker'], 'Remark': t['Remark'],
                    'Turn (timestamp) ': t['Time'], 'Turn Number': t['Turn_Number'],
                    'Episode_Shift': ai_data.get('Episode_Shift', 'SAME')
                }
                # Ensure EVERY label from prompt.csv is populated
                for item in coding_scheme:
                    row[item['label']] = ai_data.get(item['label'], "0") 
                processed_rows.append(row)
            return chunk_index, processed_rows
        except:
            time.sleep(2)
    return chunk_index, None

# ==========================================
# 4. EXECUTION & RE-INDEXING
# ==========================================

def run_coding_pipeline(vtt_file, prompt_file, template_file, output_file):
    template_cols = pd.read_csv(template_file).columns.tolist()
    merged_turns = parse_and_merge_vtt(vtt_file)
    if not merged_turns: return

    coding_scheme, instruction_text = get_coding_scheme(prompt_file)
    chunks = [merged_turns[i : i + CHUNK_SIZE] for i in range(0, len(merged_turns), CHUNK_SIZE)]
    results_map = {}

    print(f"--- Processing {vtt_file} ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, i, c, instruction_text, coding_scheme): i for i, c in enumerate(chunks)}
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Progress"):
            idx, rows = future.result()
            if rows: results_map[idx] = rows

    all_rows = []
    for i in sorted(results_map.keys()):
        all_rows.extend(results_map[i])

    # Re-indexing Episodes
    current_ep = 1
    for i, row in enumerate(all_rows):
        if i > 0 and row['Episode_Shift'] == 'NEW':
            current_ep += 1
        row['Episode Number'] = current_ep

    output_df = pd.DataFrame(all_rows)
    final_cols = [c for c in template_cols if c in output_df.columns]
    output_df[final_cols].to_csv(output_file, index=False)
    print(f"✅ Success! Results with Resolution and Episode Numbers saved to {output_file}")

if __name__ == "__main__":
    run_coding_pipeline("G30_1.vtt", "prompt.csv", "output.csv", "G30_1_Analysis.csv")