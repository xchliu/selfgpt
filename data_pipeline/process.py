"""
Data pipeline — clean and prepare corpus for training.

Steps:
  1. Load raw markdown articles
  2. Strip frontmatter, MEDIA references, HTML tags
  3. Deduplicate (remove v2/v3 duplicates)
  4. Filter by quality (min length, readability)
  5. Normalize formatting (consistent separators, clean paragraphs)
  6. Output clean corpus ready for training

Usage:
    python data_pipeline/process.py --input /path/to/gzh --output data/clean-corpus.txt
"""
import os
import re
import hashlib
import argparse


def strip_frontmatter(text):
    """Remove YAML frontmatter (--- ... ---)"""
    if text.startswith('---'):
        end = text.find('---', 3)
        if end > 0:
            return text[end + 3:].strip()
    return text.strip()


def strip_media(text):
    """Remove MEDIA: image references"""
    # Remove MEDIA:path lines
    text = re.sub(r'MEDIA:[^\n]+', '', text)
    # Remove markdown image syntax ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    return text


def strip_html_tags(text):
    """Remove HTML tags but keep content"""
    text = re.sub(r'<section[^>]*>', '', text)
    text = re.sub(r'</section>', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text


def clean_whitespace(text):
    """Normalize whitespace and line breaks"""
    # Remove trailing spaces on each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace
    return text.strip()


def get_content_hash(text):
    """Generate hash for deduplication"""
    # Normalize: lowercase, strip whitespace, remove punctuation
    normalized = re.sub(r'[\s\n，。！？、；：""''（）【】《》\-—…]', '', text.lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def is_duplicate(text, seen_hashes):
    """Check if article is duplicate based on content hash"""
    h = get_content_hash(text)
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False


def is_quality(text, min_length=200):
    """Check if article meets quality threshold"""
    if len(text) < min_length:
        return False
    # Reject if too many short lines (likely fragmented notes)
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return False
    # Reject if average line length too short (likely table/data, not prose)
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    if avg_line_len < 15:
        return False
    return True


def process_article(text):
    """Full cleaning pipeline for a single article"""
    text = strip_frontmatter(text)
    text = strip_media(text)
    text = strip_html_tags(text)
    text = clean_whitespace(text)
    return text


def process_corpus(input_dir, output_path, min_length=200):
    """
    Process all markdown files in input_dir, output clean corpus.

    Args:
        input_dir: directory containing .md files (recursive)
        output_path: output file path
        min_length: minimum article length in chars
    """
    import glob

    # Find all .md files
    files = []
    for f in glob.glob(os.path.join(input_dir, '**', '*.md'), recursive=True):
        # Skip non-article files
        if any(skip in f for skip in ['排版', 'STATUS', 'README', '.git']):
            continue
        files.append(f)

    print(f"Found {len(files)} files")

    seen_hashes = set()
    stats = {
        'total': len(files),
        'processed': 0,
        'skipped_dup': 0,
        'skipped_quality': 0,
        'skipped_short': 0,
        'total_chars': 0,
    }

    with open(output_path, 'w', encoding='utf-8') as out:
        for fpath in sorted(files):
            with open(fpath, 'r', encoding='utf-8') as f:
                raw = f.read()

            # Clean
            text = process_article(raw)

            # Dedup
            if is_duplicate(text, seen_hashes):
                stats['skipped_dup'] += 1
                continue

            # Quality filter
            if len(text) < min_length:
                stats['skipped_short'] += 1
                continue
            if not is_quality(text, min_length):
                stats['skipped_quality'] += 1
                continue

            # Write with separator
            out.write(text + '\n\n' + '=' * 50 + '\n\n')
            stats['processed'] += 1
            stats['total_chars'] += len(text)

    print(f"\n=== Results ===")
    print(f"  Total files:    {stats['total']}")
    print(f"  Processed:      {stats['processed']}")
    print(f"  Skipped (dup):  {stats['skipped_dup']}")
    print(f"  Skipped (short):{stats['skipped_short']}")
    print(f"  Skipped (qual): {stats['skipped_quality']}")
    print(f"  Output chars:   {stats['total_chars']:,}")
    print(f"  Output file:    {output_path}")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean corpus for training')
    parser.add_argument('--input', required=True, help='input directory with .md files')
    parser.add_argument('--output', default='data/clean-corpus.txt', help='output file path')
    parser.add_argument('--min_length', type=int, default=200, help='min article length')
    args = parser.parse_args()

    process_corpus(args.input, args.output, args.min_length)