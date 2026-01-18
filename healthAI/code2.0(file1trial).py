import os
import re
import json
import time
import pandas as pd
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# SETTINGS
# ============================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.1

MAX_WORKERS = 6
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2

# Round 1 (episode boundary): sliding windows
R1_WINDOW_SIZE = 80
R1_OVERLAP = 20
R1_MERGE_RULE = "majority"  # "majority" or "any"

# Round 2b (phase): windowed within episode to avoid long JSON
PH_WINDOW = 30
PH_OVERLAP = 20

# Round 3 (turn-level): windowed within episode to avoid long JSON
R3_WINDOW_SIZE = 10
R3_OVERLAP = 5

# Label sets (must match prompt.csv "Code" column exactly)
EPISODE_LEVEL_LABELS = [
    "Episode Type",
    "Episode Summary",
    "Episode Resolution",
    "Satisfaction",
    "Relief Response",
    "Action Plan",
]
PHASE_LABEL = "Phase"

TURN_LEVEL_LABELS = [
    "Turn Selection",
    "Turn Assigner",
    "Problem Addressed",
    "Behavior Encouragement",
    "Goal Push",
    "Emotion",
    "Empathy",
    "Negativity Addition",
    "Apology",
    "Reflective Question",
]

EXCLUDE_LABELS = {
    "Episode Number",
    "Turn Number",
    "Turn (timestamp) ",
    "Turn (timestamp)",
    "Time",
    "Speaker",
    "Remark",
    "Code",
    "Prompts",
    "Scale",
    "Unnamed",
}


# ============================================================
# IO: VTT -> TURNS
# ============================================================
def parse_and_merge_vtt(vtt_file_path: str):
    with open(vtt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_turns = []
    current_timestamp = None
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) -->")
    speaker_pattern = re.compile(r"^([^:\n]+):\s*(.*)")

    for line in lines:
        line = line.strip()
        time_match = time_pattern.search(line)
        if time_match:
            current_timestamp = time_match.group(1)
            continue

        speaker_match = speaker_pattern.match(line)
        if speaker_match and current_timestamp:
            raw_turns.append({
                "Time": current_timestamp,
                "Speaker": speaker_match.group(1).strip(),
                "Remark": speaker_match.group(2).strip()
            })
            current_timestamp = None

    if not raw_turns:
        return []

    merged = []
    cur = raw_turns[0].copy()
    for nxt in raw_turns[1:]:
        if nxt["Speaker"] == cur["Speaker"]:
            cur["Remark"] += " " + nxt["Remark"]
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)

    for i, t in enumerate(merged):
        t["Turn Number"] = i + 1
        t["Turn (timestamp) "] = t["Time"]
    return merged


# ============================================================
# prompt.csv -> scheme helpers
# ============================================================
def load_prompt_csv(prompt_csv_path: str):
    df = pd.read_csv(prompt_csv_path)
    scheme = []
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        if not label or label.lower().startswith("unnamed"):
            continue
        prompt = str(row.iloc[1]).strip() if len(row) > 1 else ""
        scale = str(row.iloc[2]).strip() if len(row) > 2 else ""
        scheme.append({"label": label, "prompt": prompt, "scale": scale})
    return scheme


def select_items_by_labels(full_scheme, wanted_labels):
    wanted = set(wanted_labels)
    items = []
    for it in full_scheme:
        lab = it["label"]
        if lab in EXCLUDE_LABELS:
            continue
        if lab in wanted:
            items.append(it)
    return items


def find_item_by_label(full_scheme, label):
    for it in full_scheme:
        if it["label"] == label:
            return it
    return None


def build_instruction_text(items, header="Coding Instructions"):
    txt = header + ":\n"
    for it in items:
        txt += f"- {it['label']}: {it['prompt']} (Scale: {it['scale']})\n"
    return txt


# ============================================================
# LLM call (robust JSON parse)
# ============================================================
def _extract_json_object(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def call_llm_json(system_prompt, user_prompt):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                if extracted:
                    return json.loads(extracted)
                raise
        except Exception as e:
            last_err = e
            print(f"[call_llm_json] attempt={attempt} error={type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP_SEC * attempt)

    print(f"[call_llm_json] failed after retries. last_error={type(last_err).__name__}: {last_err}")
    return None


# ============================================================
# Window builders
# ============================================================
def build_windows(seq, window_size, overlap):
    step = window_size - overlap
    out = []
    wid = 0
    n = len(seq)
    for start in range(0, n, step):
        end = min(start + window_size, n)
        out.append((wid, start, end, seq[start:end]))
        wid += 1
        if end == n:
            break
    return out


def build_episode_windows(episode_turns, window_size, overlap):
    step = window_size - overlap
    out = []
    n = len(episode_turns)
    if n <= window_size:
        return [(0, n, episode_turns)]
    for start in range(0, n, step):
        end = min(start + window_size, n)
        out.append((start, end, episode_turns[start:end]))
        if end == n:
            break
    return out


def turns_to_text(turns):
    lines = []
    for i, t in enumerate(turns, start=1):
        lines.append(f"Turn {i} | GlobalTurn {t['Turn Number']} | {t['Speaker']}: {t['Remark']}")
    return "\n".join(lines)


# ============================================================
# ROUND 1: Episode_Shift via sliding windows + merge
# ============================================================
def process_r1_window(wid, start_idx, end_idx, wturns):
    transcript = []
    for local_i, t in enumerate(wturns, start=1):
        transcript.append(
            f"LocalTurn {local_i} | GlobalTurn {t['Turn Number']} | {t['Speaker']}: {t['Remark']}"
        )
    transcript = "\n".join(transcript)

    system_prompt = (
        "You are a behavioral research assistant. "
        "ONLY task: for each turn, output Episode_Shift='NEW' if this turn STARTS a new episode "
        "(macroscopic change in purpose/goal), else 'SAME'. Be conservative. Return JSON only."
    )

    user_prompt = (
        "Return one object per turn, aligned in order.\n\n"
        f"TRANSCRIPT WINDOW:\n{transcript}\n\n"
        'Return JSON: {"results":[{"Episode_Shift":"SAME"}, ...]}\n'
    )

    data = call_llm_json(system_prompt, user_prompt)
    if not data:
        return wid, start_idx, end_idx, None

    results = data.get("results", [])
    shifts = []
    for i in range(len(wturns)):
        obj = results[i] if i < len(results) and isinstance(results[i], dict) else {}
        v = str(obj.get("Episode_Shift", "SAME")).upper().strip()
        shifts.append("NEW" if v == "NEW" else "SAME")

    shifts = (shifts + ["SAME"] * len(wturns))[:len(wturns)]
    return wid, start_idx, end_idx, shifts


def merge_episode_shifts(n_turns, window_outputs, merge_rule):
    votes = defaultdict(list)
    for start, end, shifts in window_outputs:
        if shifts is None:
            continue
        for off, v in enumerate(shifts):
            idx = start + off
            if 0 <= idx < n_turns:
                votes[idx].append(v)

    final = ["SAME"] * n_turns
    for i in range(n_turns):
        vals = votes.get(i, [])
        if not vals:
            final[i] = "SAME"
            continue
        c = Counter(vals)
        if merge_rule == "any":
            final[i] = "NEW" if c.get("NEW", 0) > 0 else "SAME"
        else:
            final[i] = "NEW" if c.get("NEW", 0) > c.get("SAME", 0) else "SAME"

    final[0] = "SAME"
    return final


def assign_episode_numbers(shifts):
    ep = 1
    ep_nums = []
    for i, s in enumerate(shifts):
        if i > 0 and s == "NEW":
            ep += 1
        ep_nums.append(ep)
    return ep_nums


# ============================================================
# ROUND 2A: Episode-level (ask once per episode)
# ============================================================
def process_episode_level(ep, ep_turns, episode_items):
    inst = build_instruction_text(episode_items, header="Episode-level Coding Scheme")
    transcript = turns_to_text(ep_turns)

    system_prompt = "You are a behavioral research assistant. Return JSON only."
    user_prompt = (
        "Answer ONCE for the whole episode.\n"
        f"{inst}\n\n"
        f"EPISODE TRANSCRIPT:\n{transcript}\n\n"
        'Return JSON: {"episode": { ... }}\n'
        "Rules: Use exact labels as keys. For binary fields output '1' or '0' (default '0')."
    )

    data = call_llm_json(system_prompt, user_prompt)
    if not data:
        return ep, None

    obj = data.get("episode", {})
    if not isinstance(obj, dict):
        obj = {}

    out = {}
    for it in episode_items:
        lab = it["label"]
        out[lab] = str(obj.get(lab, "0")).strip()
    return ep, out


# ============================================================
# ROUND 2B: Phase per turn (windowed within episode)
# ============================================================
def process_phase_window(ep, window_turns, phase_item):
    inst = build_instruction_text([phase_item], header="Phase Coding Scheme")
    transcript = turns_to_text(window_turns)

    system_prompt = "You are a behavioral research assistant. Return JSON only."
    user_prompt = (
        "Assign a Phase label to EACH turn in this window, aligned in order.\n"
        f"{inst}\n\n"
        f"TRANSCRIPT WINDOW:\n{transcript}\n\n"
        'Return JSON: {"phases": ["1","2","2", ...]}\n'
        "Rules: phases length MUST equal number of turns. Return ONLY JSON."
    )

    data = call_llm_json(system_prompt, user_prompt)
    if not data:
        return ep, None

    phases_raw = data.get("phases", [])
    phases = []
    for i in range(len(window_turns)):
        ph = phases_raw[i] if i < len(phases_raw) else "0"
        phases.append(str(ph).strip())
    phases = (phases + ["0"] * len(window_turns))[:len(window_turns)]
    return ep, phases


# ============================================================
# ROUND 3: Turn-level coding (windowed within episode + overlap merge)
# ============================================================
def process_turn_window(ep, start_local, window_turns, turn_items):
    inst = build_instruction_text(turn_items, header="Turn-level Coding Scheme")

    lines = []
    for i, t in enumerate(window_turns, start=1):
        ep_turn = start_local + i
        lines.append(
            f"LocalTurn {i} | EpTurn {ep_turn} | GlobalTurn {t['Turn Number']} | {t['Speaker']}: {t['Remark']}"
        )
    transcript = "\n".join(lines)

    system_prompt = "You are a behavioral research assistant. Return JSON only."
    user_prompt = (
        "Code EACH turn in this window using the scheme.\n"
        f"{inst}\n\n"
        f"EPISODE TURN WINDOW:\n{transcript}\n\n"
        'Return JSON: {"results": [ {...}, {...} ]}\n'
        "Rules: Use exact labels. For binary fields output '1' or '0' (default '0')."
    )

    data = call_llm_json(system_prompt, user_prompt)
    if not data:
        return start_local, None

    results = data.get("results", [])
    out = []
    for i in range(len(window_turns)):
        obj = results[i] if i < len(results) and isinstance(results[i], dict) else {}
        row = {}
        for it in turn_items:
            lab = it["label"]
            row[lab] = str(obj.get(lab, "0")).strip()
        out.append(row)

    out = (out + [{it["label"]: "0" for it in turn_items}] * len(window_turns))[:len(window_turns)]
    return start_local, out


def merge_turn_windows(ep_len, window_outputs, turn_items):
    votes = [defaultdict(list) for _ in range(ep_len)]
    for start_local, rows in window_outputs:
        if rows is None:
            continue
        for off, row in enumerate(rows):
            idx = start_local + off
            if 0 <= idx < ep_len:
                for it in turn_items:
                    lab = it["label"]
                    votes[idx][lab].append(str(row.get(lab, "0")).strip())

    merged = []
    for i in range(ep_len):
        row = {}
        for it in turn_items:
            lab = it["label"]
            vals = votes[i].get(lab, [])
            if not vals:
                row[lab] = "0"
                continue
            is_binary = all(v in ("0", "1") for v in vals)
            if is_binary:
                c = Counter(vals)
                row[lab] = "1" if c.get("1", 0) > c.get("0", 0) else "0"
            else:
                # first non-default
                pick = None
                for v in vals:
                    if v not in ("", "0", "NA", "N/A", "None"):
                        pick = v
                        break
                row[lab] = pick if pick is not None else vals[0]
        merged.append(row)
    return merged


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(vtt_file, prompt_csv, output_csv, template_csv=None):
    turns = parse_and_merge_vtt(vtt_file)
    if not turns:
        print(f"⚠️ No turns parsed from {vtt_file}")
        return

    scheme = load_prompt_csv(prompt_csv)
    episode_items = select_items_by_labels(scheme, EPISODE_LEVEL_LABELS)
    phase_item = find_item_by_label(scheme, PHASE_LABEL)
    turn_items = select_items_by_labels(scheme, TURN_LEVEL_LABELS)

    template_cols = None
    if template_csv and os.path.exists(template_csv):
        template_cols = pd.read_csv(template_csv).columns.tolist()

    # ------------------ Round 1 ------------------
    windows = build_windows(turns, R1_WINDOW_SIZE, R1_OVERLAP)
    print(f"Turns={len(turns)} | Windows={len(windows)} | window={R1_WINDOW_SIZE} overlap={R1_OVERLAP}")

    r1_outputs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for wid, start, end, wturns in windows:
            futs.append(ex.submit(process_r1_window, wid, start, end, wturns))
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Round1 windows"):
            wid, start, end, shifts = fut.result()
            r1_outputs.append((start, end, shifts))

    shifts = merge_episode_shifts(len(turns), r1_outputs, R1_MERGE_RULE)
    ep_nums = assign_episode_numbers(shifts)

    for i, t in enumerate(turns):
        t["Episode_Shift"] = shifts[i]
        t["Episode Number"] = ep_nums[i]

    episodes = defaultdict(list)
    for t in turns:
        episodes[t["Episode Number"]].append(t)

    # ------------------ Round 2A ------------------
    print("--- Round 2A: Episode-level ---")
    r2_ep_fields = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for ep, ep_turns in episodes.items():
            futs.append(ex.submit(process_episode_level, ep, ep_turns, episode_items))
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Round2A episodes"):
            ep, fields = fut.result()
            if fields is None:
                print(f"⚠️ Round2A failed at Episode {ep} (defaulting to zeros)")
                fields = {it["label"]: "0" for it in episode_items}
            r2_ep_fields[ep] = fields

    # ------------------ Round 2B ------------------
    print("--- Round 2B: Phase (windowed) ---")
    r2_phase = {}  # (ep, global_turn) -> phase
    for ep in tqdm(sorted(episodes.keys()), desc="Round2B per-episode"):
        ep_turns = episodes[ep]
        wins = build_episode_windows(ep_turns, PH_WINDOW, PH_OVERLAP)

        phase_votes = [[] for _ in range(len(ep_turns))]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = []
            for start_local, end_local, slice_turns in wins:
                futs.append(ex.submit(process_phase_window, ep, slice_turns, phase_item))
            for fut, (start_local, end_local, slice_turns) in zip(futs, wins):
                _, phases = fut.result()
                if phases is None:
                    phases = ["0"] * len(slice_turns)
                for off, ph in enumerate(phases):
                    idx = start_local + off
                    if 0 <= idx < len(ep_turns):
                        phase_votes[idx].append(ph)

        merged_phases = []
        for vals in phase_votes:
            if not vals:
                merged_phases.append("0")
            else:
                merged_phases.append(Counter(vals).most_common(1)[0][0])

        for i, t in enumerate(ep_turns):
            r2_phase[(ep, t["Turn Number"])] = merged_phases[i]

    # ------------------ Round 3 ------------------
    print("--- Round 3: Turn-level (windowed) ---")
    r3_turn = {}  # (ep, global_turn) -> {turn-level fields}

    for ep in tqdm(sorted(episodes.keys()), desc="Round3 per-episode"):
        ep_turns = episodes[ep]
        ep_len = len(ep_turns)
        wins = build_episode_windows(ep_turns, R3_WINDOW_SIZE, R3_OVERLAP)

        window_outputs = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = []
            for start_local, end_local, slice_turns in wins:
                futs.append(ex.submit(process_turn_window, ep, start_local, slice_turns, turn_items))
            for fut in as_completed(futs):
                start_local, rows = fut.result()
                window_outputs.append((start_local, rows))

        merged = merge_turn_windows(ep_len, window_outputs, turn_items)
        for i, t in enumerate(ep_turns):
            r3_turn[(ep, t["Turn Number"])] = merged[i]

    # ------------------ Combine ------------------
    rows = []
    for t in turns:
        ep = t["Episode Number"]
        gturn = t["Turn Number"]

        row = {
            "Time": t["Time"],
            "Speaker": t["Speaker"],
            "Remark": t["Remark"],
            "Turn (timestamp) ": t["Turn (timestamp) "],
            "Turn Number": gturn,
            "Episode_Shift": t["Episode_Shift"],
            "Episode Number": ep,
            "Phase": r2_phase.get((ep, gturn), "0"),
        }

        # Episode-level fields
        for k, v in r2_ep_fields.get(ep, {it["label"]: "0" for it in episode_items}).items():
            row[k] = v

        # Turn-level fields
        for k, v in r3_turn.get((ep, gturn), {it["label"]: "0" for it in turn_items}).items():
            row[k] = v

        rows.append(row)

    out_df = pd.DataFrame(rows)
    if template_cols:
        final_cols = [c for c in template_cols if c in out_df.columns]
        extras = [c for c in out_df.columns if c not in final_cols]
        out_df = out_df[final_cols + extras]

    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")


if __name__ == "__main__":
    run_pipeline(
        vtt_file="G30_1.vtt",
        prompt_csv="prompt.csv",
        template_csv="output.csv",          # optional
        output_csv="G30_1_Analysis_2.0.csv",
    )
