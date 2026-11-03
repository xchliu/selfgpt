"""
Auto-generate SFT Q&A pairs from articles.

Multi-strategy extraction:
  1. Title → Question, intro → Answer
  2. Section heading → Question, section content → Answer
  3. Key sentence → Question, surrounding context → Answer
  4. "为什么" sentences → Question, following explanation → Answer
  5. Blockquote → wisdom quote, generate Q&A around it
  6. Multi-turn: follow-up questions from article content
  7. Contrasting viewpoints → debate-style Q&A

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
    """Extract title from article"""
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
        line_stripped = line.strip()
        if line_stripped.startswith('## '):
            if current_heading and current_content:
                sections.append((current_heading, '\n'.join(current_content).strip()))
            current_heading = line_stripped[3:].strip()
            current_content = []
        elif current_heading:
            current_content.append(line)
        elif line_stripped and not line_stripped.startswith('#'):
            # content before first ## heading
            pass

    if current_heading and current_content:
        sections.append((current_heading, '\n'.join(current_content).strip()))

    return sections


def extract_paragraphs(article):
    """Extract all substantive paragraphs"""
    paragraphs = []
    current = []

    for line in article.split('\n'):
        line = line.strip()
        if not line:
            if current:
                para = '\n'.join(current).strip()
                if len(para) > 30:
                    paragraphs.append(para)
                current = []
        else:
            current.append(line)

    if current:
        para = '\n'.join(current).strip()
        if len(para) > 30:
            paragraphs.append(para)

    return paragraphs


def extract_blockquotes(article):
    """Extract blockquote content"""
    quotes = []
    for line in article.split('\n'):
        line = line.strip()
        if line.startswith('> ') and len(line) > 10:
            quote = line[2:].strip()
            if len(quote) > 15:
                quotes.append(quote)
    return quotes


def clean_text(text):
    """Remove markdown formatting"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^> ?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate_answer(text, max_chars=400):
    """Truncate to sentence boundary"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for i in range(len(truncated) - 1, max(0, len(truncated) - 80), -1):
        if truncated[i] in '。！？.!\n':
            return truncated[:i + 1]
    return truncated + '...'


def make_qa(question, answer, min_answer_len=40):
    """Create a Q&A pair if quality is sufficient"""
    answer = clean_text(answer)
    answer = truncate_answer(answer)
    question = question.strip().rstrip('？?')
    if len(answer) >= min_answer_len and len(question) >= 5:
        return {
            "conversations": [
                {"role": "user", "content": question + "？"},
                {"role": "assistant", "content": answer}
            ]
        }
    return None


# --- Strategies ---

def strategy_title_intro(article):
    """Strategy 1: Title → Q, intro paragraphs → A"""
    qa = []
    title = extract_title(article)
    if not title:
        return qa

    paragraphs = extract_paragraphs(article)
    if paragraphs:
        # Use first 1-2 paragraphs as answer
        answer = '\n'.join(paragraphs[:2])
        pair = make_qa(title, answer)
        if pair:
            qa.append(pair)

    return qa


def strategy_sections(article):
    """Strategy 2: Each ## heading → Q, section content → A"""
    qa = []
    sections = extract_sections(article)

    for heading, content in sections:
        if len(content) < 40:
            continue
        pair = make_qa(heading, content)
        if pair:
            qa.append(pair)

    return qa


def strategy_key_sentences(article):
    """Strategy 3: Find key declarative sentences, turn into Q&A"""
    qa = []
    paragraphs = extract_paragraphs(article)

    for para in paragraphs:
        para_clean = clean_text(para)
        # Split into sentences
        sentences = re.split(r'[。！？\n]', para_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        for i, sentence in enumerate(sentences):
            # If sentence makes a claim, use it as answer, ask "what about X"
            if len(sentence) > 30 and len(sentence) < 150:
                # Generate a question from the sentence
                # Pattern: extract subject/topic from sentence
                question = generate_question_from_sentence(sentence)
                if question:
                    # Answer = this sentence + next sentence for context
                    if i + 1 < len(sentences):
                        answer = sentence + '。' + sentences[i + 1]
                    else:
                        answer = sentence + '。'
                    pair = make_qa(question, answer, min_answer_len=30)
                    if pair:
                        qa.append(pair)

    return qa


def generate_question_from_sentence(sentence):
    """Try to generate a question from a declarative sentence"""
    # Pattern 1: "X是Y" → "什么是X？"
    m = re.match(r'^([^.，,]{2,15})是', sentence)
    if m:
        return f"什么是{m.group(1)}"

    # Pattern 2: "X不是Y" → "X是什么？"
    m = re.match(r'^([^.，,]{2,15})不是', sentence)
    if m:
        return f"{m.group(1)}是什么"

    # Pattern 3: "X应该/可以/需要Y" → "X应该怎么做？"
    m = re.match(r'^([^.，,]{2,15})(应该|可以|需要|要)', sentence)
    if m:
        return f"{m.group(1)}应该怎么做"

    # Pattern 4: "为什么X" already a question → use directly
    if sentence.startswith('为什么') or sentence.startswith('为何'):
        return sentence[:50]

    # Pattern 5: "关键在于X" → "关键是什么？"
    if '关键' in sentence[:10]:
        return "关键是什么"

    return None


def strategy_why_questions(article):
    """Strategy 4: Find '因为' explanations, generate why questions"""
    qa = []
    paragraphs = extract_paragraphs(article)

    for para in paragraphs:
        para_clean = clean_text(para)
        # Find "因为...所以..." or "X的原因是Y" patterns
        if '因为' in para_clean or '原因是' in para_clean:
            # Find the sentence before 因为 as the question context
            sentences = re.split(r'[。！？]', para_clean)
            for i, sent in enumerate(sentences):
                if '因为' in sent and i > 0:
                    prev = sentences[i - 1].strip()
                    if len(prev) > 10:
                        question = prev[:50] + "为什么？"
                        answer = sent.strip()
                        if len(answer) > 30:
                            pair = make_qa(question, answer, min_answer_len=30)
                            if pair:
                                qa.append(pair)
                                break  # one per paragraph

    return qa


def strategy_blockquotes(article):
    """Strategy 5: Blockquotes → wisdom Q&A"""
    qa = []
    quotes = extract_blockquotes(article)
    title = extract_title(article)

    for quote in quotes:
        quote_clean = clean_text(quote)
        if len(quote_clean) > 20:
            # Generate question about the quote
            if title:
                question = f"{title}中最重要的一句话是什么？"
            else:
                question = "最重要的一句话是什么？"
            answer = quote_clean + "\n\n这句话说出了一个核心道理。"
            pair = make_qa(question, answer, min_answer_len=20)
            if pair:
                qa.append(pair)
                break  # one per article

    return qa


def strategy_multi_turn(article):
    """Strategy 6: Generate multi-turn conversations"""
    qa = []
    sections = extract_sections(article)
    title = extract_title(article)

    if not title or len(sections) < 2:
        return qa

    # First turn: title question, first section answer
    conversations = []

    # Turn 1
    first_q = title + "？"
    first_a = clean_text(sections[0][1])
    first_a = truncate_answer(first_a, max_chars=200)
    if len(first_a) < 30:
        return qa

    conversations.append({"role": "user", "content": first_q})
    conversations.append({"role": "assistant", "content": first_a})

    # Turn 2: follow-up about next section
    if len(sections) > 1:
        follow_q = sections[1][0] + "？"
        follow_a = clean_text(sections[1][1])
        follow_a = truncate_answer(follow_a, max_chars=200)
        if len(follow_a) >= 30:
            conversations.append({"role": "user", "content": follow_q})
            conversations.append({"role": "assistant", "content": follow_a})

    if len(conversations) >= 4:
        qa.append({"conversations": conversations})

    return qa


def generate_qa_from_article(article):
    """Run all strategies on one article"""
    qa = []

    qa.extend(strategy_title_intro(article))
    qa.extend(strategy_sections(article))
    qa.extend(strategy_key_sentences(article))
    qa.extend(strategy_why_questions(article))
    qa.extend(strategy_blockquotes(article))
    qa.extend(strategy_multi_turn(article))

    return qa


def generate_sft_data(corpus_path, output_path, max_pairs=500):
    """Generate SFT data with multi-strategy extraction"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    articles = split_articles(text)
    print(f"Articles: {len(articles)}")

    all_qa = []
    strategy_counts = {
        'title_intro': 0,
        'sections': 0,
        'key_sentences': 0,
        'why_questions': 0,
        'blockquotes': 0,
        'multi_turn': 0,
    }

    for article in articles:
        # Count by strategy
        title_qa = strategy_title_intro(article)
        strategy_counts['title_intro'] += len(title_qa)

        section_qa = strategy_sections(article)
        strategy_counts['sections'] += len(section_qa)

        key_qa = strategy_key_sentences(article)
        strategy_counts['key_sentences'] += len(key_qa)

        why_qa = strategy_why_questions(article)
        strategy_counts['why_questions'] += len(why_qa)

        quote_qa = strategy_blockquotes(article)
        strategy_counts['blockquotes'] += len(quote_qa)

        multi_qa = strategy_multi_turn(article)
        strategy_counts['multi_turn'] += len(multi_qa)

        all_qa.extend(title_qa)
        all_qa.extend(section_qa)
        all_qa.extend(key_qa)
        all_qa.extend(why_qa)
        all_qa.extend(quote_qa)
        all_qa.extend(multi_qa)

    print(f"\n=== Strategy breakdown ===")
    for name, count in strategy_counts.items():
        print(f"  {name}: {count}")

    print(f"\nTotal generated: {len(all_qa)}")

    # Deduplicate by question content
    seen_questions = set()
    unique_qa = []
    for qa in all_qa:
        q = qa['conversations'][0]['content']
        if q not in seen_questions:
            seen_questions.add(q)
            unique_qa.append(qa)

    print(f"After dedup: {len(unique_qa)}")

    # Shuffle and limit
    random.seed(42)
    random.shuffle(unique_qa)
    unique_qa = unique_qa[:max_pairs]

    # Merge with hand-written data
    existing_path = os.path.join(os.path.dirname(output_path), 'sft-data.json')
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Hand-written pairs: {len(existing)}")
        unique_qa = existing + unique_qa

    print(f"Total output: {len(unique_qa)}")

    # Stats
    multi_turn_count = sum(1 for qa in unique_qa if len(qa['conversations']) > 2)
    print(f"Multi-turn conversations: {multi_turn_count}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_qa, f, ensure_ascii=False, indent=2)

    print(f"Output: {output_path}")

    # Print samples by strategy
    print("\n=== Samples ===")
    for qa in unique_qa[:5]:
        conv = qa['conversations']
        print(f"Q: {conv[0]['content']}")
        print(f"A: {conv[1]['content'][:100]}...")
        if len(conv) > 2:
            print(f"  [multi-turn: {len(conv)} messages]")
        print()

    return unique_qa


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(base, 'data', 'clean-corpus.txt')
    output = os.path.join(base, 'data', 'sft', 'sft-auto.json')
    generate_sft_data(corpus, output, max_pairs=500)