"""
Extract plain text from Wikipedia JSON for pretraining.

Input: data/raw/wikipedia-zh-cn-20260501.json (1.5M articles, 807M chars)
Output: data/corpus/wiki-clean.txt (max_chars specified)

Each article: {"id": "...", "title": "...", "tags": "...", "text": "..."}
We extract: title + text, separated by newline.
"""
import json
import os
import re


def clean_wiki_text(text):
    """Clean Wikipedia markup residue"""
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove references [...]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove wiki links [[target|display]] -> display
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_wikipedia(input_path, output_path, max_chars=10_000_000):
    """Extract plain text from Wikipedia JSON"""
    total = 0
    kept = 0
    skipped = 0
    output = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if total >= max_chars:
                break

            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = obj.get('text', '')
            title = obj.get('title', '')

            # Skip short articles
            if len(text) < 100:
                skipped += 1
                continue

            # Clean
            text = clean_wiki_text(text)

            # Combine: title + text
            article = f"{title}\n{text}\n"

            if len(article) < 150:
                skipped += 1
                continue

            output.append(article)
            total += len(article)
            kept += 1

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"=== Wikipedia Extraction ===")
    print(f"  Articles kept:   {kept:,}")
    print(f"  Articles skipped: {skipped:,}")
    print(f"  Total chars:     {total:,}")
    print(f"  Avg length:      {total // kept if kept else 0:,}")
    print(f"  Output:          {output_path}")

    # Sample
    print(f"\n=== Sample ===")
    print(output[0][:300] if output else "No data")

    return kept


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, 'data', 'raw', 'wikipedia-zh-cn-20260501.json')
    output_path = os.path.join(base, 'data', 'corpus', 'wiki-clean.txt')
    extract_wikipedia(input_path, output_path, max_chars=10_000_000)