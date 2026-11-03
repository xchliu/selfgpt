"""
Auto-generate SFT Q&A pairs from articles.

Reads clean corpus, extracts key paragraphs, generates question-answer pairs.

Strategies:
  1. Title → Question, first 2-3 paragraphs → Answer
  2. Section heading → Question, section content → Answer
  3. Key sentence → Question, surrounding context → Answer

Output: data/sft/sft-auto.json
"""
import os
import re
import json
import random


def split_articles(text):
    """Split corpus into individual articles by separator"""
    parts = text.split('=' * 50)
    articles = []
    for part in parts:
        part = part.strip()
        if len(part) > 100:
            articles.append(part)
    return articles


def extract_title(article):
    """Extract title from article (first # heading or first line)"""
    lines = article.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
        if line and not line.startswith('>') and not line.startswith('**'):
            return line.strip()
    return None


def extract_sections(article):
    """Extract sections by ## headings"""
    sections = []
    current_heading = None
    current_content = []

    for line in article.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            if current_heading and current_content:
                sections.append((current_heading, '\n'.join(current_content).strip()))
            current_heading = line[3:].strip()
            current_content = []
        elif current_heading:
            current_content.append(line)

    if current_heading and current_content:
        sections.append((current_heading, '\n'.join(current_content).strip()))

    return sections


def clean_text(text):
    """Remove markdown formatting, keep plain text"""
    # Remove bold markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove inline code
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Remove blockquotes markers
    text = re.sub(r'^> ?', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate_answer(text, max_chars=300):
    """Truncate answer to max_chars, ending at sentence boundary"""
    if len(text) <= max_chars:
        return text

    # Find sentence boundary near max_chars
    truncated = text[:max_chars]
    # Try to end at a period/question mark/exclamation
    for i in range(len(truncated) - 1, max(0, len(truncated) - 50), -1):
        if truncated[i] in '。！？.!?\n':
            return truncated[:i + 1]

    return truncated + '...'


def generate_qa_from_article(article):
    """Generate Q&A pairs from a single article"""
    qa_pairs = []
    title = extract_title(article)

    if not title:
        return qa_pairs

    sections = extract_sections(article)

    # Strategy 1: Title → Question, intro → Answer
    lines = article.strip().split('\n')
    intro_lines = []
    for line in lines[1:]:  # skip title
        line = line.strip()
        if line.startswith('## '):
            break
        if line and len(line) > 10:
            intro_lines.append(line)
        if len(intro_lines) >= 3:
            break

    if intro_lines:
        answer = clean_text('\n'.join(intro_lines))
        answer = truncate_answer(answer)
        if len(answer) > 30:
            qa_pairs.append({
                "conversations": [
                    {"role": "user", "content": title + "？"},
                    {"role": "assistant", "content": answer}
                ]
            })

    # Strategy 2: Section headings → Questions
    for heading, content in sections:
        content_clean = clean_text(content)
        if len(content_clean) < 30:
            continue

        answer = truncate_answer(content_clean)
        if len(answer) > 30:
            qa_pairs.append({
                "conversations": [
                    {"role": "user", "content": heading + "？"},
                    {"role": "assistant", "content": answer}
                ]
            })

    return qa_pairs


def generate_sft_data(corpus_path, output_path, max_pairs=200):
    """
    Generate SFT data from clean corpus.

    Args:
        corpus_path: path to clean corpus
        output_path: output JSON path
        max_pairs: maximum number of Q&A pairs to generate
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    articles = split_articles(text)
    print(f"Articles: {len(articles)}")

    all_qa = []
    for article in articles:
        qa = generate_qa_from_article(article)
        all_qa.extend(qa)

    print(f"Generated Q&A pairs: {len(all_qa)}")

    # Shuffle and limit
    random.seed(42)
    random.shuffle(all_qa)
    all_qa = all_qa[:max_pairs]

    # Merge with existing hand-written data if present
    existing_path = os.path.join(os.path.dirname(output_path), 'sft-data.json')
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Existing hand-written pairs: {len(existing)}")
        all_qa = existing + all_qa

    print(f"Total Q&A pairs: {len(all_qa)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"Output: {output_path}")

    # Print samples
    print("\n=== Samples ===")
    for qa in all_qa[:3]:
        conv = qa['conversations']
        print(f"Q: {conv[0]['content']}")
        print(f"A: {conv[1]['content'][:80]}...")
        print()

    return all_qa


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(base, 'data', 'clean-corpus.txt')
    output = os.path.join(base, 'data', 'sft', 'sft-auto.json')
    generate_sft_data(corpus, output, max_pairs=200)