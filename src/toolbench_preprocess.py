"""ToolBench preprocessing: raw answer trajectories -> conversations.json.

Faithful port of the abandoned turn-encoding repo's
work/scripts/preprocess_toolbench.py (commit pinned in
KIT_PROVENANCE.json as toolbench_turn_encoding_commit), with one
deliberate change per the zone-risk charter's data amendment
(2026-07-23): the stated goal (answer_generation.query) becomes a
first-class `goal_text` field on every conversation record instead of
surviving only as the first user turn.

Reads data/raw/toolbench/answer/G{1,2,3}_answer/*.json (fetched at the
pinned HF revision via `python main.py --fetch-data`), writes
data/raw/toolbench/conversations.json. Success = valid_data AND
finish_type == "give_answer"; the longest train_messages chain is the
solution path.

    python toolbench_preprocess.py
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "toolbench"
ANSWER = RAW / "answer"
OUT = RAW / "conversations.json"


def iter_raw(max_files=None):
    count = 0
    for group in ["G1_answer", "G2_answer", "G3_answer"]:
        gdir = ANSWER / group
        if not gdir.exists():
            print(f"  Warning: {gdir} not found, skipping")
            continue
        files = sorted(f for f in os.listdir(gdir) if f.endswith(".json"))
        print(f"  {group}: {len(files)} files")
        for fname in files:
            try:
                with open(gdir / fname) as f:
                    data = json.load(f)
                data["_source_file"] = fname
                data["_group"] = group
                yield data
                count += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if max_files and count >= max_files:
                return


def parse_assistant_message(content):
    """ReAct-style assistant message -> thought + action turns."""
    turns = []
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", content,
                              re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        if thought:
            turns.append({"turn_type": "assistant", "content": thought,
                          "metadata": {"step_type": "thought"}})

    action_match = re.search(r"Action:\s*(.*?)(?=\nAction Input:|$)", content,
                             re.DOTALL)
    action_input_match = re.search(r"Action Input:\s*(.*?)$", content,
                                   re.DOTALL)
    if action_match:
        action_name = action_match.group(1).strip()
        action_input = (action_input_match.group(1).strip()
                        if action_input_match else "")
        if action_name.lower() == "finish":
            return turns
        params = {}
        if action_input:
            try:
                params = json.loads(action_input)
                if not isinstance(params, dict):
                    params = {"input": params}
            except json.JSONDecodeError:
                params = {"raw": action_input}
        service = action_name
        if "_for_" in action_name:
            service = action_name.split("_for_")[-1]
        turns.append({
            "turn_type": "tool_call",
            "content": (f"{action_name}({json.dumps(params)})" if params
                        else action_name),
            "metadata": {"tool_name": action_name, "service": service,
                         "parameters": params}})

    if not turns and content.strip():
        turns.append({"turn_type": "assistant", "content": content.strip(),
                      "metadata": {}})
    return turns


def convert_trajectory(raw):
    ag = raw.get("answer_generation", {})
    if not ag:
        return None
    valid_data = ag.get("valid_data", False)
    finish_type = ag.get("finish_type", "")
    query = ag.get("query", "")
    chains = ag.get("train_messages", [])
    if not chains or not query:
        return None

    chain = max(chains, key=len) if isinstance(chains[0], list) else chains
    success = valid_data and finish_type == "give_answer"

    turns = []
    tool_call_count = 0
    seen_user_query = False
    for msg in chain:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        function_call = msg.get("function_call")

        if role == "system":
            continue
        elif role == "user":
            if not seen_user_query:
                turns.append({"turn_type": "user", "content": content.strip(),
                              "metadata": {}})
                seen_user_query = True
            else:
                turns.append({"turn_type": "user",
                              "content": content.strip()[:500],
                              "metadata": {"retry": True}})
        elif role == "assistant":
            if function_call:
                tool_name = function_call.get("name", "")
                args_str = function_call.get("arguments", "")
                if tool_name.lower() == "finish":
                    continue
                params = {}
                if args_str:
                    try:
                        params = json.loads(args_str)
                        if not isinstance(params, dict):
                            params = {"input": params}
                    except json.JSONDecodeError:
                        params = {"raw": args_str[:200]}
                service = tool_name
                if "_for_" in tool_name:
                    service = tool_name.split("_for_")[-1]
                if content and content.strip():
                    turns.append({"turn_type": "assistant",
                                  "content": content.strip()[:500],
                                  "metadata": {"step_type": "thought"}})
                turns.append({
                    "turn_type": "tool_call",
                    "content": (f"{tool_name}({json.dumps(params)})"
                                if params else tool_name),
                    "metadata": {"tool_name": tool_name, "service": service,
                                 "parameters": params, "success": success}})
                tool_call_count += 1
            elif content and content.strip():
                parsed = parse_assistant_message(content)
                if parsed:
                    for turn in parsed:
                        if turn["turn_type"] == "tool_call":
                            tool_call_count += 1
                            turn["metadata"]["success"] = success
                        turns.append(turn)
                else:
                    turns.append({"turn_type": "assistant",
                                  "content": content.strip()[:500],
                                  "metadata": {}})
        elif role == "function":
            result_content = content if content else ""
            if len(result_content) > 500:
                result_content = result_content[:500] + "..."
            turns.append({"turn_type": "tool_result",
                          "content": result_content, "metadata": {}})

    if len(turns) < 2 or tool_call_count == 0:
        return None

    services = list({t["metadata"]["service"] for t in turns
                     if t["turn_type"] == "tool_call"
                     and "service" in t["metadata"]})
    return {
        "dialogue_id": raw.get("_source_file", "unknown").replace(".json", ""),
        "goal_text": query,
        "services": services,
        "turns": turns,
        "split": "train",
        "has_failure": not success,
        "n_turns": len(turns),
        "source": "toolbench",
        "group": raw.get("_group", "unknown"),
        "valid_data": valid_data,
        "finish_type": finish_type,
    }


def run(max_files=None):
    if not ANSWER.exists():
        raise FileNotFoundError(
            f"{ANSWER} missing - fetch first: python main.py --fetch-data")
    conversations = []
    for raw in iter_raw(max_files):
        conv = convert_trajectory(raw)
        if conv:
            conversations.append(conv)

    OUT.write_text(json.dumps(conversations))
    groups = Counter(c["group"] for c in conversations)
    n_fail = sum(1 for c in conversations if c["has_failure"])
    res = {"n_conversations": len(conversations),
           "failure_rate": n_fail / max(len(conversations), 1),
           "groups": dict(groups),
           "with_goal_text": sum(1 for c in conversations if c["goal_text"]),
           "avg_turns": (sum(c["n_turns"] for c in conversations)
                         / max(len(conversations), 1))}
    save_result("toolbench_preprocess", res, dataset="toolbench",
                params={"max_files": max_files})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
