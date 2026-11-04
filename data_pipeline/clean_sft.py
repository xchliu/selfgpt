"""
Clean SFT data: filter bad Q&A, normalize punctuation, remove noise.

Input: data/sft/sft-final.json
Output: data/sft/sft-clean.json

Checks:
  1. Question quality: meaningful, not too short/long, natural
  2. Answer quality: coherent, proper length, has substance
  3. Punctuation: normalize ？。！，etc.
  4. Remove:重复, 无效闲聊, HTML/markdown残留, 乱码
  5. Remove: 问答不相关 (keyword overlap check)
"""
import os
import re
import json


def normalize_punctuation(text):
    """Normalize punctuation"""
    # Full-width to half-width for English punctuation
    replacements = {
        '？': '?', '！': '!', '：': ':', '；': ';',
        '（': '(', '）': ')', '【': '[', '】': ']',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '…': '...', '——': '--',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # But keep Chinese punctuation for Chinese text
    # Actually, keep full-width for Chinese sentences
    # Revert: keep Chinese style punctuation
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    text = text.replace(':', '：')
    text = text.replace(';', '；')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    
    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove spaces around Chinese punctuation
    text = re.sub(r'\s+([，。！？；：、）」』])', r'\1', text)
    text = re.sub(r'([（「『])\s+', r'\1', text)
    
    # Normalize multiple punctuation
    text = re.sub(r'。{2,}', '。', text)
    text = re.sub(r'？{2,}', '？', text)
    text = re.sub(r'！{2,}', '！', text)
    text = re.sub(r'，{2,}', '，', text)
    
    return text.strip()


def remove_noise(text):
    """Remove HTML, markdown residue, control characters"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove markdown image
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # Remove MEDIA references
    text = re.sub(r'MEDIA:[^\s]*', '', text)
    # Remove separator lines
    text = re.sub(r'={5,}', '', text)
    # Remove excessive dashes
    text = re.sub(r'-{3,}', '', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Remove replacement character
    text = text.replace('\ufffd', '')
    return text.strip()


def is_good_question(q):
    """Check if question is meaningful and natural"""
    q = q.strip()
    
    # Length check
    if len(q) < 5 or len(q) > 80:
        return False
    
    # Must end with ? or ？
    if not q.endswith('？') and not q.endswith('?'):
        return False
    
    # Remove trailing ? for content check
    content = q.rstrip('？?').strip()
    if len(content) < 4:
        return False
    
    # Reject fragments
    bad_starts = ['而且', '或者', '但是', '所以', '因为', '另外', '同时',
                  '其实', '不过', '因此', '后来', '还是', '也是', '不是',
                  '没有', '有的', '有些', '可能', '也许', '大约', '比如',
                  '例如', '首先', '其次', '最后', '然而', '尽管', '无论',
                  '只要', '只有', '除非', '一旦', '即使', '虽然', '反正',
                  '总之', '一般来说', '通常', '往往', '似乎', '稍微']
    for bad in bad_starts:
        if content.startswith(bad):
            return False
    
    # Reject if too many special characters
    special = sum(1 for c in content if not c.isalnum() and c not in '？?_-' and '\u4e00' > c or c > '\u9fff')
    if special > len(content) * 0.3:
        return False
    
    # Must have Chinese or meaningful English
    cjk = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
    alpha = sum(1 for c in content if c.isalpha())
    if cjk + alpha < len(content) * 0.4:
        return False
    
    # Reject pure number questions
    if re.match(r'^[\d\s？?]+$', q):
        return False
    
    # Reject questions that are just one word + ?
    if len(content.replace('？', '').replace('?', '').strip()) < 4:
        return False
    
    return True


def is_good_answer(a):
    """Check if answer is coherent and substantive"""
    a = a.strip()
    
    # Length check
    if len(a) < 30 or len(a) > 600:
        return False
    
    # Must have Chinese content
    cjk = sum(1 for c in a if '\u4e00' <= c <= '\u9fff')
    if cjk < len(a) * 0.15:
        return False
    
    # Reject if too many special characters / symbols
    special = sum(1 for c in a if c in '|=+*#<>[]{}\\/')
    if special > len(a) * 0.1:
        return False
    
    # Reject if mostly numbers/table
    digits = sum(1 for c in a if c.isdigit())
    if digits > len(a) * 0.3:
        return False
    
    # Reject if has too many line breaks (likely table/list)
    if a.count('\n') > 5:
        return False
    
    # Must end with proper punctuation
    if a[-1] not in '。！？.!?\n':
        return False
    
    # Reject if contains placeholder text
    placeholders = ['XXX', 'xxx', 'TODO', 'todo', 'TBD', 'tbd', 'N/A', 'n/a']
    for p in placeholders:
        if p in a:
            return False
    
    return True


def is_relevant(question, answer):
    """Check if answer is relevant to question (keyword overlap)"""
    q_content = question.rstrip('？?').strip()
    
    # Extract keywords from question (2+ char Chinese segments)
    q_keywords = set()
    # Extract meaningful Chinese words
    for i in range(len(q_content)):
        for j in range(i+2, min(i+6, len(q_content)+1)):
            word = q_content[i:j]
            if all('\u4e00' <= c <= '\u9fff' for c in word):
                q_keywords.add(word)
    
    if not q_keywords:
        return True  # Can't check, allow it
    
    # Check if any keyword appears in answer
    for kw in q_keywords:
        if kw in answer:
            return True
    
    # If no keyword overlap, check if question type matches answer
    if '什么' in question and len(answer) > 50:
        return True  # Definition questions are usually relevant
    
    return False


def clean_conversation(conv):
    """Clean a full conversation"""
    cleaned = []
    for msg in conv:
        role = msg['role']
        content = msg.get('content', '')
        
        # Remove noise
        content = remove_noise(content)
        # Normalize punctuation
        content = normalize_punctuation(content)
        
        if len(content) < 2:
            return None  # Bad conversation
        
        cleaned.append({"role": role, "content": content})
    
    return cleaned if len(cleaned) >= 2 else None


def clean_sft_data(input_path, output_path):
    """Main cleaning pipeline"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Input: {len(data)} conversations")
    
    stats = {
        'total': len(data),
        'kept': 0,
        'bad_question': 0,
        'bad_answer': 0,
        'not_relevant': 0,
        'clean_failed': 0,
        'too_short': 0,
    }
    
    cleaned_data = []
    seen_questions = set()
    
    for item in data:
        conv = item.get('conversations', [])
        
        # Clean conversation
        conv = clean_conversation(conv)
        if conv is None:
            stats['clean_failed'] += 1
            continue
        
        # Validate each Q&A pair
        valid = True
        for i in range(0, len(conv) - 1, 2):
            user_msg = conv[i]
            assistant_msg = conv[i + 1]
            
            if user_msg['role'] != 'user' or assistant_msg['role'] != 'assistant':
                valid = False
                break
            
            q = user_msg['content']
            a = assistant_msg['content']
            
            # Check quality
            if not is_good_question(q):
                stats['bad_question'] += 1
                valid = False
                break
            
            if not is_good_answer(a):
                stats['bad_answer'] += 1
                valid = False
                break
            
            # Check relevance (only for single-turn)
            if len(conv) == 2 and not is_relevant(q, a):
                stats['not_relevant'] += 1
                valid = False
                break
        
        if not valid:
            continue
        
        # Dedup by first question
        first_q = conv[0]['content']
        if first_q in seen_questions:
            continue
        seen_questions.add(first_q)
        
        # Check total length
        total_len = sum(len(m['content']) for m in conv)
        if total_len < 40:
            stats['too_short'] += 1
            continue
        
        cleaned_data.append({"conversations": conv})
        stats['kept'] += 1
    
    print(f"\n=== Cleaning Results ===")
    print(f"  Total input:     {stats['total']:,}")
    print(f"  Kept:            {stats['kept']:,}")
    print(f"  Bad question:    {stats['bad_question']:,}")
    print(f"  Bad answer:      {stats['bad_answer']:,}")
    print(f"  Not relevant:    {stats['not_relevant']:,}")
    print(f"  Clean failed:    {stats['clean_failed']:,}")
    print(f"  Too short:       {stats['too_short']:,}")
    print(f"  Dedup removed:   {stats['total'] - stats['kept'] - stats['bad_question'] - stats['bad_answer'] - stats['not_relevant'] - stats['clean_failed'] - stats['too_short']:,}")
    
    # Stats on output
    single = sum(1 for d in cleaned_data if len(d['conversations']) == 2)
    multi = sum(1 for d in cleaned_data if len(d['conversations']) > 2)
    avg_len = sum(len(m['content']) for d in cleaned_data for m in d['conversations']) / len(cleaned_data) if cleaned_data else 0
    
    print(f"\n=== Output Stats ===")
    print(f"  Single-turn: {single:,}")
    print(f"  Multi-turn:  {multi:,}")
    print(f"  Avg msg length: {avg_len:.0f} chars")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Output: {output_path}")
    
    # Samples
    print("\n=== Samples ===")
    for d in cleaned_data[:5]:
        conv = d['conversations']
        print(f"  [{len(conv)} turns]")
        for msg in conv[:2]:
            print(f"    {msg['role']}: {msg['content'][:80]}")
        print()
    
    return cleaned_data


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, 'data', 'sft', 'sft-final.json')
    output_path = os.path.join(base, 'data', 'sft', 'sft-clean.json')
    clean_sft_data(input_path, output_path)