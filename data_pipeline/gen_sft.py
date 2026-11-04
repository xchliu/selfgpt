"""
Auto-generate SFT Q&A pairs from articles — v2.

Improvements:
  - Better question generation (什么是/为什么/怎么做/核心区别)
  - Quality filtering: reject bad questions
  - More multi-turn conversations (2-3 turns)
  - Answer完整性: keep full logic blocks
  - Dedup by question
"""
import os
import re
import json
import random


def split_articles(text):
    parts = text.split('=' * 50)
    return [p.strip() for p in parts if len(p.strip()) > 100]


def extract_title(article):
    for line in article.strip().split('\n'):
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
        if line and not line.startswith('>') and not line.startswith('**'):
            return line.strip()
    return None


def extract_sections(article):
    sections = []
    current_heading = None
    current_content = []
    for line in article.split('\n'):
        line_s = line.strip()
        if line_s.startswith('## '):
            if current_heading and current_content:
                sections.append((current_heading, '\n'.join(current_content).strip()))
            current_heading = line_s[3:].strip()
            current_content = []
        elif current_heading:
            current_content.append(line)
    if current_heading and current_content:
        sections.append((current_heading, '\n'.join(current_content).strip()))
    return sections


def extract_paragraphs(article):
    paragraphs = []
    current = []
    for line in article.split('\n'):
        line = line.strip()
        if not line:
            if current:
                para = '\n'.join(current).strip()
                if len(para) > 40:
                    paragraphs.append(para)
                current = []
        else:
            current.append(line)
    if current:
        para = '\n'.join(current).strip()
        if len(para) > 40:
            paragraphs.append(para)
    return paragraphs


def extract_blockquotes(article):
    quotes = []
    for line in article.split('\n'):
        line = line.strip()
        if line.startswith('> ') and len(line) > 15:
            quotes.append(line[2:].strip())
    return quotes


def clean_text(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^> ?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_good_question(q):
    q = q.strip()
    if len(q) < 5 or len(q) > 60:
        return False
    bad_starts = ['而且', '或者', '但是', '所以', '因为', '另外', '同时', '其实',
                  '不过', '因此', '后来', '以前', '还是', '也是', '不是',
                  '没有', '有的', '有些', '可能', '也许', '大约']
    for bad in bad_starts:
        if q.startswith(bad):
            return False
    cjk = sum(1 for c in q if '\u4e00' <= c <= '\u9fff')
    if cjk < 3:
        return False
    return True


def is_good_answer(a):
    a = a.strip()
    if len(a) < 40 or len(a) > 800:
        return False
    cjk = sum(1 for c in a if '\u4e00' <= c <= '\u9fff')
    if cjk < len(a) * 0.2:
        return False
    return True


def truncate_at_sentence(text, max_chars=400):
    if len(text) <= max_chars:
        return text
    for i in range(max_chars, max(0, max_chars - 100), -1):
        if i < len(text) and text[i] in '。！？':
            return text[:i + 1]
    return text[:max_chars] + '...'


def make_qa(question, answer):
    answer = clean_text(answer)
    answer = truncate_at_sentence(answer)
    question = question.strip().rstrip('？?')
    question = re.sub(r'[#*`]', '', question).strip()
    if is_good_question(question + '？') and is_good_answer(answer):
        return {
            "conversations": [
                {"role": "user", "content": question + "？"},
                {"role": "assistant", "content": answer}
            ]
        }
    return None


# --- Strategies ---

def strategy_title_intro(article):
    qa = []
    title = extract_title(article)
    if not title:
        return qa
    paragraphs = extract_paragraphs(article)
    if paragraphs:
        answer = '\n'.join(paragraphs[:2])
        pair = make_qa(title, answer)
        if pair:
            qa.append(pair)
    return qa


def strategy_sections(article):
    qa = []
    for heading, content in extract_sections(article):
        if len(content) < 50:
            continue
        pair = make_qa(heading, content)
        if pair:
            qa.append(pair)
    return qa


def strategy_what_is(article):
    """Find 'X是Y' → '什么是X？'"""
    qa = []
    paragraphs = extract_paragraphs(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        sentences = re.split(r'[。！？\n]', para_clean)
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            m = re.match(r'^([^\s，,。]{2,12})是(.{20,})', sent)
            if m:
                topic = m.group(1)
                if topic in ['这', '那', '我', '你', '他', '她', '它', '我们', '他们', '什么', '为', '但', '所']:
                    continue
                question = f"什么是{topic}"
                answer = sent + '。'
                if i + 1 < len(sentences) and len(sentences[i + 1].strip()) > 10:
                    answer += sentences[i + 1].strip() + '。'
                pair = make_qa(question, answer)
                if pair:
                    qa.append(pair)
    return qa


def strategy_why(article):
    """Find '因为/原因是/关键在于' → '为什么'"""
    qa = []
    paragraphs = extract_paragraphs(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        if '因为' not in para_clean and '原因是' not in para_clean and '关键' not in para_clean:
            continue
        sentences = re.split(r'[。！？]', para_clean)
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            if '因为' in sent or '原因是' in sent or '关键在于' in sent:
                if i > 0:
                    prev = sentences[i - 1].strip()
                    if len(prev) > 10:
                        question = prev[:40] + "为什么"
                        answer = sent + '。'
                        pair = make_qa(question, answer)
                        if pair:
                            qa.append(pair)
                            break
    return qa


def strategy_how_to(article):
    """Find '应该/需要/可以' → '怎么做'"""
    qa = []
    paragraphs = extract_paragraphs(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        sentences = re.split(r'[。！？]', para_clean)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            m = re.match(r'^([^\s，,。]{2,15})(应该|需要|可以|必须|要)(.{15,})', sent)
            if m:
                topic = m.group(1)
                if topic in ['这', '那', '我', '你', '他', '我们', '什么']:
                    continue
                question = f"{topic}应该怎么做"
                answer = sent + '。'
                pair = make_qa(question, answer)
                if pair:
                    qa.append(pair)
    return qa


def strategy_blockquotes(article):
    qa = []
    quotes = extract_blockquotes(article)
    title = extract_title(article)
    for quote in quotes[:1]:
        quote_clean = clean_text(quote)
        if len(quote_clean) > 20:
            question = f"{title}中最重要的一句话是什么" if title else "核心观点是什么"
            pair = make_qa(question, quote_clean)
            if pair:
                qa.append(pair)
    return qa


def strategy_multi_turn(article):
    """2-3 turn conversations from sections"""
    qa = []
    sections = extract_sections(article)
    title = extract_title(article)
    if not title or len(sections) < 2:
        return qa

    conversations = []
    first_a = clean_text(sections[0][1])
    first_a = truncate_at_sentence(first_a, max_chars=250)
    if len(first_a) < 40:
        return qa
    conversations.append({"role": "user", "content": title + "？"})
    conversations.append({"role": "assistant", "content": first_a})

    if len(sections) > 1:
        s2_q = sections[1][0]
        s2_a = clean_text(sections[1][1])
        s2_a = truncate_at_sentence(s2_a, max_chars=250)
        if len(s2_a) >= 40 and is_good_question(s2_q + "？"):
            conversations.append({"role": "user", "content": s2_q + "？"})
            conversations.append({"role": "assistant", "content": s2_a})

    if len(sections) > 2:
        s3_q = sections[2][0]
        s3_a = clean_text(sections[2][1])
        s3_a = truncate_at_sentence(s3_a, max_chars=250)
        if len(s3_a) >= 40 and is_good_question(s3_q + "？"):
            conversations.append({"role": "user", "content": s3_q + "？"})
            conversations.append({"role": "assistant", "content": s3_a})

    if len(conversations) >= 4:
        qa.append({"conversations": conversations})
    return qa


def strategy_comparison(article):
    """Find '不是...而是...' → comparison"""
    qa = []
    paragraphs = extract_paragraphs(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        m = re.search(r'不是(.{5,50})[，,]而是(.{10,})', para_clean)
        if m:
            pair = make_qa("核心区别是什么", para_clean[:300])
            if pair:
                qa.append(pair)
                break
    return qa


def strategy_conclusion(article):
    """Last paragraph → '总结/结论' question"""
    qa = []
    paragraphs = extract_paragraphs(article)
    title = extract_title(article)
    if not paragraphs or len(paragraphs) < 3:
        return qa
    # Last 1-2 paragraphs often contain conclusion
    last_para = clean_text(paragraphs[-1])
    if len(last_para) > 50:
        question = f"{title}的结论是什么" if title else "结论是什么"
        pair = make_qa(question, last_para)
        if pair:
            qa.append(pair)
    return qa


def strategy_list_items(article):
    """Find numbered/bulleted lists → individual Q&A"""
    qa = []
    lines = article.split('\n')
    current_item = None
    current_text = []
    
    for line in lines:
        line_s = line.strip()
        # Match: "1. xxx" or "- xxx" or "* xxx" or "第一，xxx"
        m = re.match(r'^(?:\d+[.、]|[•\-\*]|第[一二三四五六七八九十]+[，,])\s*(.+)', line_s)
        if m:
            if current_item and current_text:
                answer = clean_text('\n'.join(current_text))
                if len(answer) > 40:
                    pair = make_qa(current_item, answer)
                    if pair:
                        qa.append(pair)
            current_item = m.group(1)[:50]
            current_text = [m.group(1)]
        elif current_item and line_s:
            current_text.append(line_s)
    
    if current_item and current_text:
        answer = clean_text('\n'.join(current_text))
        if len(answer) > 40:
            pair = make_qa(current_item, answer)
            if pair:
                qa.append(pair)
    
    # Limit to avoid too many from one article
    return qa[:5]


def strategy_data_points(article):
    """Find sentences with numbers/percentages → '数据是多少'"""
    qa = []
    paragraphs = extract_paragraphs(article)
    title = extract_title(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        # Find sentences with percentages or large numbers
        sentences = re.split(r'[。！？]', para_clean)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            # Has percentage or large number
            if re.search(r'\d+%|\d+万|\d+亿|\d+,\d{3}', sent):
                question = f"{title}的数据说明了什么" if title else "数据说明了什么"
                answer = sent + '。'
                pair = make_qa(question, answer)
                if pair:
                    qa.append(pair)
                    break  # one per paragraph
    return qa[:3]  # max 3 per article


def strategy_paragraph_summary(article):
    """Each substantive paragraph → '关于什么' question"""
    qa = []
    paragraphs = extract_paragraphs(article)
    title = extract_title(article)
    
    for i, para in enumerate(paragraphs[2:], start=2):  # skip first 2 (covered by title_intro)
        para_clean = clean_text(para)
        if len(para_clean) < 60:
            continue
        
        # Extract first sentence as topic
        first_sent = para_clean.split('。')[0].strip()
        if len(first_sent) < 10:
            continue
        
        # Generate question from first sentence
        # Try "X的核心是" pattern
        m = re.match(r'(.{5,30})的核心', first_sent)
        if m:
            question = f"{m.group(1)}的核心是什么"
        else:
            # Use paragraph position as question context
            question = f"{title}的要点" if title else "要点是什么"
        
        answer = truncate_at_sentence(para_clean, max_chars=300)
        pair = make_qa(question, answer)
        if pair:
            qa.append(pair)
    
    return qa[:3]  # max 3 per article


def strategy_definition(article):
    """Find 'X是指/Y叫做X' patterns → definition Q&A"""
    qa = []
    paragraphs = extract_paragraphs(article)
    for para in paragraphs:
        para_clean = clean_text(para)
        sentences = re.split(r'[。！？]', para_clean)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            # "X是指Y" or "Y叫做X" or "X意味着"
            m = re.match(r'^([^\s，,。]{2,12})(是指|意味着|指的是)(.{15,})', sent)
            if m:
                topic = m.group(1)
                if topic in ['这', '那', '它', '什么']:
                    continue
                question = f"什么是{topic}"
                pair = make_qa(question, sent + '。')
                if pair:
                    qa.append(pair)
                    continue
            # "叫做X" pattern
            m = re.match(r'(.{5,30})叫做([^\s，,。]{2,12})', sent)
            if m:
                topic = m.group(2)
                question = f"什么是{topic}"
                pair = make_qa(question, sent + '。')
                if pair:
                    qa.append(pair)
    return qa


def generate_qa_from_article(article):
    qa = []
    qa.extend(strategy_title_intro(article))
    qa.extend(strategy_sections(article))
    qa.extend(strategy_what_is(article))
    qa.extend(strategy_why(article))
    qa.extend(strategy_how_to(article))
    qa.extend(strategy_blockquotes(article))
    qa.extend(strategy_multi_turn(article))
    qa.extend(strategy_comparison(article))
    qa.extend(strategy_conclusion(article))
    qa.extend(strategy_list_items(article))
    qa.extend(strategy_data_points(article))
    qa.extend(strategy_paragraph_summary(article))
    qa.extend(strategy_definition(article))
    return qa


def generate_sft_data(corpus_path, output_path, max_pairs=2000):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    articles = split_articles(text)
    print(f"Articles: {len(articles)}")

    all_qa = []
    for article in articles:
        qa = generate_qa_from_article(article)
        all_qa.extend(qa)

    single_turn = sum(1 for q in all_qa if len(q['conversations']) == 2)
    multi_turn = sum(1 for q in all_qa if len(q['conversations']) > 2)

    print(f"\n=== Breakdown ===")
    print(f"  Single-turn: {single_turn}")
    print(f"  Multi-turn:  {multi_turn}")
    print(f"  Total:       {len(all_qa)}")

    seen = set()
    unique = []
    for qa in all_qa:
        q = qa['conversations'][0]['content']
        if q not in seen:
            seen.add(q)
            unique.append(qa)
    print(f"After dedup: {len(unique)}")

    random.seed(42)
    random.shuffle(unique)
    unique = unique[:max_pairs]

    existing_path = os.path.join(os.path.dirname(output_path), 'sft-data.json')
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Hand-written: {len(existing)}")
        unique = existing + unique

    multi_count = sum(1 for q in unique if len(q['conversations']) > 2)
    print(f"Total output: {len(unique)} (multi-turn: {multi_count})")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"Output: {output_path}")

    print("\n=== Samples ===")
    for qa in unique[:5]:
        conv = qa['conversations']
        print(f"Q: {conv[0]['content']}")
        print(f"A: {conv[1]['content'][:100]}...")
        if len(conv) > 2:
            print(f"  [multi-turn: {len(conv)} msgs]")
        print()

    return unique


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(base, 'data', 'clean-corpus.txt')
    output = os.path.join(base, 'data', 'sft', 'sft-auto.json')
    generate_sft_data(corpus, output, max_pairs=2000)