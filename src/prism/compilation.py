import re
from collections import defaultdict
import numpy as np
from .retrieve import raw_records

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'what', 'where', 'when', 'who', 'whom',
    'whose', 'which', 'why', 'how', 'that', 'this', 'these', 'those', 'it', 'its',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
    'into', 'over', 'after', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
    'while', 'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 'there',
    'wow', 'hey', 'hi', 'hello', 'yeah', 'yes', 'yep', 'no', 'nope', 'oh', 'ah',
    'awesome', 'cool', 'great', 'nice', 'good', 'well', 'really', 'totally',
    'definitely', 'absolutely', 'sure', 'thanks', 'thank', 'glad', 'happy',
    'sounds', 'sound', 'looks', 'look', 'like', 'know', 'think', 'get', 'got',
    'going', 'go', 'went', 'gone', 'make', 'made', 'take', 'took', 'taken',
    'see', 'saw', 'seen', 'come', 'came', 'want', 'wanted', 'need', 'needed',
    'way', 'thing', 'things', 'time', 'times', 'day', 'days', 'lot', 'lots',
    'much', 'many', 'some', 'any', 'all', 'both', 'each', 'more', 'most', 'other',
    'such', 'only', 'own', 'same', 'also', 'back', 'even', 'still', 'tell', 'told',
    'say', 'said', 'ask', 'asked', 'talk', 'talking', 'talked', 'hear', 'heard',
    'feel', 'feeling', 'felt', 'keep', 'kept', 'let', 'put', 'seem', 'seems',
    'try', 'tried', 'trying', 'call', 'called', 'find', 'found', 'give', 'gave',
    'recorded', 'speaker', 'listener', 'conv', 'pm', 'am'
}

def porter_lite_stem(w: str) -> str:
    'Purely algorithmic English suffix stripper (inflectional morphology only, zero domain dictionary).'
    if len(w) <= 3:
        return w
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('ing') and len(w) > 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in 'aeiou':
            return base[:-1]
        if base.endswith(('at', 'iz', '', 'v', 'k')) or len(base) == 3 and base[1] in 'aeiou' and base[2] not in 'aeiouwxy':
            return base
        return base
    if w.endswith('ed') and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in 'aeiou':
            return base[:-1]
        return base
    if w.endswith('es') and len(w) > 4 and not w.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
        return w[:-1]
    if w.endswith('s') and not w.endswith(('ss', 'us', 'is')) and len(w) > 3:
        return w[:-1]
    return w

def clean_tokenize_algorithmic(text: str, speaker_only: str = "") -> list[str]:
    text_no_meta = re.sub(
        r'\[(?:conv-[^\]]+|Speaker:[^\]]+|Date:[^\]]+|Recorded:[^\]]+|Image search query[^\]]+|Visual description[^\]]+)\]',
        ' ', text
    )
    words = re.findall(r'[a-zA-Z0-9]+', text_no_meta.lower())
    tokens = []
    for w in words:
        if w in STOPWORDS:
            continue
        stem = porter_lite_stem(w)
        if stem in STOPWORDS:
            continue
        tokens.append(stem)
        if stem != w:
            tokens.append(w)
    if speaker_only:
        sp = speaker_only.lower()
        tokens.extend([sp, sp])
    return tokens if tokens else ['_empty']

def build_clean_compiled_turn(turns, i, turn_facts_list):
    '100% general conversational de-contextualization + 14B Stage-1 Fact compilation.'
    t = turns[i]
    sp = t['speaker']
    txt = t['text']

    txt_b = re.sub(r'\bI\b', sp, txt)
    txt_b = re.sub(r"\bI'm\b", f"{sp} is", txt_b)
    txt_b = re.sub(r"\bI've\b", f"{sp} has", txt_b)
    txt_b = re.sub(r"\bI'd\b", f"{sp} would", txt_b)
    txt_b = re.sub(r"\bI'll\b", f"{sp} will", txt_b)
    txt_b = re.sub(r'\b[Mm]y\b', f"{sp}'s", txt_b)
    txt_b = re.sub(r'\b[Mm]e\b', sp, txt_b)


    prev_ctx = ""
    if i > 0 and turns[i - 1]['session'] == t['session']:
        pt = turns[i - 1]
        if '?' in pt['text']:
            q_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', pt['text']) if '?' in s]
            if q_sents:
                qs = " ".join(q_sents[-2:])
                qs = re.sub(r'\b[Yy]ou\b', sp, qs)
                qs = re.sub(r'\b[Yy]our\b', f"{sp}'s", qs)
                prev_ctx = f"[Answering: {qs}] "


    vis_parts = []
    if t.get('caption'):
        vis_parts.append(t['caption'])
    if t.get('image_query'):
        vis_parts.append(t['image_query'])
    vis_str = f" [Visual: {' | '.join(vis_parts)}]" if vis_parts else ""


    facts_str = ""
    if turn_facts_list:
        facts_str = " [Facts: " + " | ".join(turn_facts_list) + "]"
    return f"[{t['id']}] [Speaker: {sp}] [Date: {t['recorded_at']}] {prev_ctx}{sp}: {txt_b}{vis_str}{facts_str}"

def number(session):
    return int(session.split('_')[1])

def fact_records(conversation, checkpoints):
    'The screened FACT representation, with stable chronological ordering.'
    known = {t['id']: number(t['session']) for t in conversation['turns']}
    rows = []
    for checkpoint in sorted(checkpoints, key=lambda c: number(c['session'])):
        limit = number(checkpoint['session'])
        for fact in checkpoint['facts']:
            ids = fact['evidence_ids']
            if not ids or any(i not in known or known[i] > limit for i in ids):
                raise ValueError('unknown or future fact source')
            if not any(known[i] == limit for i in ids):
                raise ValueError('no current fact source')
            text = fact['text']
            if fact.get('time_expression'):
                text += ' [original time expression: ' + fact['time_expression'] + ']'
            rows.append({'id': 'atomic:' + conversation['id'] + ':' + checkpoint['session'] + ':' + fact['id'],
                         'kind': 'detail', 'subject': '', 'text': text,
                         'evidence_ids': list(ids), 'group': ''})
    return rows

def compile_multiview_for_cid(cid, actor, fact_recs, fact_vecs, org_recs, org_vecs, emb):
    turns = actor['turns']
    raw_recs = raw_records(actor)
    tids = [r['id'] for r in raw_recs]
    tid_to_idx = {tid: i for i, tid in enumerate(tids)}

    turn_facts_text = defaultdict(list)
    turn_fact_idxs = defaultdict(list)
    for fi, frec in enumerate(fact_recs):
        ftxt = re.sub(r'^\[[^\]]+\]\s*', '', frec['text'])
        for ev in frec.get('evidence_ids', []):
            key = ev if ev in tid_to_idx else f"{cid}:{ev}"
            if key in tid_to_idx:
                ti = tid_to_idx[key]
                turn_facts_text[ti].append(ftxt)
                turn_fact_idxs[ti].append(fi)

    turn_org_text = defaultdict(list)
    turn_org_idxs = defaultdict(list)
    for oi, orec in enumerate(org_recs):
        otxt = orec['text']
        for ev in orec.get('evidence_ids', []):
            key = ev if ev in tid_to_idx else f"{cid}:{ev}"
            if key in tid_to_idx:
                ti = tid_to_idx[key]
                turn_org_text[ti].append(otxt)
                turn_org_idxs[ti].append(oi)


    clean_texts = [build_clean_compiled_turn(turns, i, turn_facts_text[i]) for i in range(len(turns))]
    clean_vecs = np.asarray(emb.embed(clean_texts, 'clean_final_smc_tmc_emb'), dtype=np.float32)
    clean_vecs /= np.maximum(np.linalg.norm(clean_vecs, axis=1, keepdims=True), 1e-12)


    bm25_raw_tokens = [re.findall(r'\w+', x.lower()) or ['_'] for x in clean_texts]
    bm25_stem_tokens = [
        clean_tokenize_algorithmic(clean_texts[i] + " " + " ".join(turn_org_text[i]), speaker_only=turns[i]['speaker'])
        for i in range(len(turns))
    ]
    return {
        'cid': cid,
        'tids': tids,
        'clean_vecs': clean_vecs.tolist(),
        'org_vecs': org_vecs.tolist(),
        'turn_fact_idxs': {str(k): v for k, v in turn_fact_idxs.items()},
        'turn_org_idxs': {str(k): v for k, v in turn_org_idxs.items()},
        'bm25_raw_tokens': bm25_raw_tokens,
        'bm25_stem_tokens': bm25_stem_tokens,
    }
