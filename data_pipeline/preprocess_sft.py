"""
Preprocess SFT JSONL corpus: filter, sample, merge with existing Q&A.

Input: data/raw/sft_t2t_mini.jsonl (905K conversations)
Output: data/sft/sft-final.json (merged SFT data)

Steps:
  1. Filter: quality check, remove reasoning_content, min length
  2. Sample: take N conversations (full corpus too large)
  3. Merge: combine with auto-generated Q&A from articles
  4. Output: unified JSON format
"""
import os
import json
import random


def clean_conversation(conv):
    """Clean a conversation: remove reasoning_content, trim"""
    cleaned = []
    for msg in conv:
        role = msg.get('role', '')
        content = msg.get('content', '')
        # Skip empty or too short messages
        if len(content) < 5:
            continue
        # Remove reasoning_content field if present
        msg_clean = {"role": role, "content": content}
        cleaned.append(msg_clean)
    return cleaned


def is_quality(conv):
    """Quality filter"""
    if len(conv) < 2:
        return False
    # At least one assistant response with substance
    has_good_response = False
    for msg in conv:
        if msg['role'] == 'assistant' and len(msg['content']) > 20:
            has_good_response = True
            break
    if not has_good_response:
        return False
    # Reject if too many turns (context too long for our model)
    if len(conv) > 10:
        return False
    # Total content length check
    total = sum(len(m['content']) for m in conv)
    if total > 1000 or total < 50:
        return False
    return True


def preprocess_sft_jsonl(input_path, output_path, max_samples=8000):
    """Process large SFT JSONL into clean subset"""
    processed = 0
    skipped_quality = 0
    skipped_short = 0
    total = 0

    print(f"Input: {input_path}")
    print(f"Max samples: {max_samples:,}")
    print(f"Processing...")

    results = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(results) >= max_samples:
                break
            total += 1

            try:
                obj = json.loads(line.strip())
                conv = obj.get('conversations', [])
            except (json.JSONDecodeError, KeyError):
                continue

            # Clean
            conv = clean_conversation(conv)

            if len(conv) < 2:
                skipped_short += 1
                continue
            if not is_quality(conv):
                skipped_quality += 1
                continue

            results.append({"conversations": conv})
            processed += 1

            if processed % 5000 == 0:
                print(f"  {processed:,} processed, {total:,} read")

    print(f"\n=== Results ===")
    print(f"  Total read:      {total:,}")
    print(f"  Processed:       {processed:,}")
    print(f"  Skipped (short): {skipped_short:,}")
    print(f"  Skipped (qual):  {skipped_quality:,}")

    # Merge with existing auto-generated Q&A
    auto_path = os.path.join(os.path.dirname(output_path), 'sft-auto.json')
    if os.path.exists(auto_path):
        with open(auto_path, 'r', encoding='utf-8') as f:
            auto = json.load(f)
        print(f"  Auto-generated:  {len(auto):,}")
        results = auto + results

    # Stats
    multi_turn = sum(1 for r in results if len(r['conversations']) > 2)
    print(f"  Total output:    {len(results):,} (multi-turn: {multi_turn})")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  Output:          {output_path}")

    # Samples
    print("\n=== Samples ===")
    for r in results[:3]:
        conv = r['conversations']
        print(f"  [{len(conv)} turns]")
        for msg in conv[:4]:
            print(f"    {msg['role']}: {msg['content'][:60]}")
        print()

    return results


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, 'data', 'raw', 'sft_t2t_mini.jsonl')
    output_path = os.path.join(base, 'data', 'sft', 'sft-final.json')
    preprocess_sft_jsonl(input_path, output_path, max_samples=8000)