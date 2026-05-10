import os
import re
import json
from pathlib import Path

from agent import get_reply

# =========================================================
# CONFIG
# =========================================================

TRACE_DIR = "sample_conversations"

# =========================================================
# HELPERS
# =========================================================

def extract_turns(markdown_text: str):

    """
    Extract user/assistant turns from markdown traces.
    """

    lines = markdown_text.splitlines()

    turns = []

    current_role = None
    current_content = []

    for line in lines:

        line = line.rstrip()

        # USER
        if line.strip() == "**User**":

            # flush previous
            if current_role and current_content:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })

            current_role = "user"
            current_content = []

            continue

        # AGENT
        if line.strip() == "**Agent**":

            if current_role and current_content:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })

            current_role = "assistant"
            current_content = []

            continue

        # QUOTED USER TEXT
        if current_role == "user":

            if line.strip().startswith(">"):
                current_content.append(
                    line.replace(">", "", 1).strip()
                )

        # AGENT TEXT
        elif current_role == "assistant":

            # skip metadata lines
            if (
                line.startswith("_`") or
                line.startswith("|") or
                line.startswith("---")
            ):
                continue

            current_content.append(line)

    # flush final
    if current_role and current_content:
        turns.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })

    return turns


def clean_trace_turns(turns):

    """
    Remove evaluator metadata and empty turns.
    """

    cleaned = []

    for t in turns:

        content = t["content"].strip()

        if not content:
            continue

        # remove markdown artifacts
        content = re.sub(r"_`.*?`_", "", content)
        content = re.sub(r"\|.*?\|", "", content)

        cleaned.append({
            "role": t["role"],
            "content": content.strip()
        })

    return cleaned


def print_recommendations(recs):

    if not recs:
        print("Recommendations: []")
        return

    print("Recommendations:")

    for r in recs:

        print(
            f"  - [{r['test_type']}] "
            f"{r['name']}"
        )


# =========================================================
# RUN SINGLE TRACE
# =========================================================

def run_trace(trace_path):

    print("\n")
    print("=" * 80)
    print(f"TRACE: {trace_path.name}")
    print("=" * 80)

    with open(trace_path, "r", encoding="utf-8") as f:
        markdown = f.read()

    turns = extract_turns(markdown)
    turns = clean_trace_turns(turns)

    messages = []

    turn_num = 0

    hallucination_detected = False

    for turn in turns:

        # only simulate USER turns
        if turn["role"] != "user":
            continue

        turn_num += 1

        print("\n")
        print("-" * 80)
        print(f"TURN {turn_num}")
        print("-" * 80)

        print(f"\nUSER:\n{turn['content']}")

        messages.append({
            "role": "user",
            "content": turn["content"]
        })

        try:

            result = get_reply(messages)

        except Exception as e:

            print("\nERROR:")
            print(str(e))

            continue

        reply = result.get("reply", "")
        recs = result.get("recommendations", [])
        end_flag = result.get("end_of_conversation", False)

        print(f"\nASSISTANT:\n{reply}\n")

        print_recommendations(recs)

        print(f"\nend_of_conversation: {end_flag}")

        # hallucination detection
        for r in recs:

            if (
                "name" not in r or
                "url" not in r or
                "test_type" not in r
            ):
                hallucination_detected = True

        # append assistant response
        messages.append({
            "role": "assistant",
            "content": reply
        })

    print("\n")
    print("=" * 80)

    if hallucination_detected:
        print("STATUS: POSSIBLE HALLUCINATION DETECTED")
    else:
        print("STATUS: PASSED BASIC VALIDATION")

    print("=" * 80)


# =========================================================
# MAIN
# =========================================================

def main():

    trace_dir = Path(TRACE_DIR)

    if not trace_dir.exists():

        print(f"Directory not found: {TRACE_DIR}")
        return

    files = sorted(trace_dir.glob("*.md"))

    if not files:

        print("No markdown traces found.")
        return

    print("\n")
    print("=" * 80)
    print(f"FOUND {len(files)} TRACE FILES")
    print("=" * 80)

    for file in files:

        run_trace(file)


if __name__ == "__main__":

    main()