"""
Preprocess large JSONL corpus into clean training text.

v2: Split multi-QA entries into individual answer blocks.

Input: pretrain_t2t_mini.jsonl
Output: data/pretrain-clean.txt

Each JSONL entry contains multiple Q&A crammed together:
  "给我写首诗。<诗歌内容>。请描述天气。<天气描述>。那穿什么？<穿衣建议>"

This script splits them and keeps only the answer portions.
"""
import os
import sys
import json
import re
import argparse


def split_qa(text):
    """
    Split multi-QA text into individual answer segments.
    """
    # Remove initial prompt
    text = re.sub(r'^(给我|帮我|请|写一个|根据|从这个|提取|生成|请写|请生成|请帮|请根据|描述一下|解释|怎样|如何|以|为|将|使用|对话模拟|将以下|请根据)[^。？]*[。？\n]', '', text)

    # Split on follow-up question markers
    split_patterns = [
        r'(?<=[。！？])\s*(?:好的[，,。]?\s*(?:那|请|我|现在|能否|给我|帮我))',
        r'(?<=[。！？])\s*(?:明白了[，,]?\s*(?:那|请|我))',
        r'(?<=[。！？])\s*(?:那(?:我|请|你|能否|给我|帮我))',
        r'(?<=[。！？])\s*(?:请问)',
        r'(?<=[。！？])\s*(?:能否)',
        r'(?<=[。！？])\s*(?:给我)',
        r'(?<=[。！？])\s*(?:帮我)',
        r'(?<=[。！？])\s*(?:请(?:给我|帮我|描述|解释|写|生成|告诉|根据))',
        r'(?<=[。！？])\s*(?:根据以下)',
        r'(?<=[。！？])\s*(?:写一个)',
        r'(?<=[。！？])\s*(?:描述一下)',
        r'(?<=[。！？])\s*(?:如何)',
    ]

    segments = [text]
    for pattern in split_patterns:
        new_segments = []
        for seg in segments:
            parts = re.split(pattern, seg)
            new_segments.extend(parts)
        segments = new_segments

    # Clean each segment - remove leading question text
    answers = []
    for seg in segments:
        seg = seg.strip()
        # Remove leading question (text before first 。if short)
        seg = re.sub(r'^[^。]{2,30}[？?]', '', seg)
        seg = seg.strip()
        if len(seg) >= 30:
            answers.append(seg)

    return answers if answers else [text.strip()] if len(text.strip()) >= 30 else []


def clean_text(text):
    """Basic cleaning"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_quality(text, min_length=50, max_length=1500):
    """Quality filter"""
    if len(text) < min_length or len(text) > max_length:
        return False
    if len(set(text)) < len(text) * 0.1:
        return False
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cjk_count < len(text) * 0.15:
        return False
    return True


def preprocess_jsonl(input_path, output_path, max_chars=5000000, min_length=50):
    """Process JSONL into clean text corpus - v2 with QA splitting"""
    total_lines = 0
    processed = 0
    skipped_quality = 0
    skipped_short = 0
    total_chars = 0
    qa_split_count = 0

    print(f"Input: {input_path}")
    print(f"Max chars: {max_chars:,}")
    print(f"Processing...")

    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                if total_chars >= max_chars:
                    break

                total_lines += 1

                try:
                    obj = json.loads(line.strip())
                    text = obj.get('text', '')
                except (json.JSONDecodeError, KeyError):
                    continue

                # Split multi-QA into individual answers
                answers = split_qa(text)
                if len(answers) > 1:
                    qa_split_count += 1

                for answer in answers:
                    if total_chars >= max_chars:
                        break

                    answer = clean_text(answer)

                    if len(answer) < min_length:
                        skipped_short += 1
                        continue
                    if not is_quality(answer):
                        skipped_quality += 1
                        continue

                    fout.write(answer + '\n\n' + '=' * 50 + '\n\n')
                    processed += 1
                    total_chars += len(answer)

                if processed % 10000 == 0:
                    print(f"  {processed:,} entries, {total_chars:,} chars, {qa_split_count:,} split")

    print(f"\n=== Results ===")
    print(f"  Total lines read:  {total_lines:,}")
    print(f"  Processed:         {processed:,}")
    print(f"  QA splits:         {qa_split_count:,}")
    print(f"  Skipped (short):   {skipped_short:,}")
    print(f"  Skipped (quality): {skipped_quality:,}")
    print(f"  Output chars:      {total_chars:,}")
    print(f"  Output file:       {output_path}")

    return total_chars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess JSONL corpus v2')
    parser.add_argument('--input', default='data/raw/pretrain_t2t_mini.jsonl')
    parser.add_argument('--output', default='data/corpus/pretrain-clean.txt')
    parser.add_argument('--max_chars', type=int, default=5000000)
    parser.add_argument('--min_length', type=int, default=50)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(base, args.output) if not os.path.isabs(args.output) else args.output

    preprocess_jsonl(input_path, output_path, args.max_chars, args.min_length)