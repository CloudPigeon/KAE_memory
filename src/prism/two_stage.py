"""Two-stage memory compilation with per-fact recall and source budgets."""
import json
import numpy as np
from rank_bm25 import BM25Okapi

from .data import render_turn
from .io import parse_json
from .memory import Archive
from .prompts import RESOURCE_DIR
from .retrieve import memory_records, tokens

MEMORY_BUDGET = 4000
SOURCE_BUDGET = 3000
PER_FACT_LIMIT = 8
KINDS = ('events', 'updates', 'details', 'loops')


def load_prompts():
    return {name: {'text': (RESOURCE_DIR / filename).read_text(encoding='utf-8')}
            for name, filename in [('facts', 'FACT_EXTRACT.en.txt'), ('organize', 'MEMORY_ORGANIZE.en.txt')]}

def session_no(session_id):
    return int(str(session_id).split('_')[1])

def visible_turns(conversation, session_id):
    'Turns the incremental writer may read: current session and earlier only.'
    limit = session_no(session_id)
    return [t for t in conversation['turns'] if session_no(t['session']) <= limit]

def session_text(conversation, session):
    byid = {t['id']: t for t in conversation['turns']}
    return '\n'.join(render_turn(byid[i]) for i in session['source_ids'])

def reference_context(conversation, session, limit=4):
    'Short preceding context, enough to resolve references, not to re-extract.'
    ids = [s['id'] for s in conversation['sessions']]
    at = ids.index(session['id'])
    if at == 0:
        return ''
    previous = conversation['sessions'][at - 1]
    byid = {t['id']: t for t in conversation['turns']}
    turns = [byid[i] for i in previous['source_ids']][-limit:]
    return '\n'.join(render_turn(t) for t in turns)

def facts_prompt(prompts, session, reference=''):
    text = prompts['facts']['text'].rstrip('\n')
    return (text + '\n\nSESSION:\n' + session
            + '\n\nREFERENCE_CONTEXT:\n' + (reference or '(none)'))

def validate_facts(payload, conversation, session):
    'Accept only the three contracted fields with real, non-future sources.'
    if not isinstance(payload, dict) or not isinstance(payload.get('facts'), list):
        raise ValueError('facts stage must return {"facts": [...]}')
    known = {t['id'] for t in conversation['turns']}
    current = set(session['source_ids'])
    limit = session_no(session['id'])
    facts, rejected = [], []
    for index, item in enumerate(payload['facts']):
        if not isinstance(item, dict):
            rejected.append({'index': index, 'reason': 'not an object', 'item': item})
            continue
        text = item.get('text')
        when = item.get('time_expression', '')
        ids = item.get('evidence_ids')
        if not isinstance(text, str) or not text.strip():
            rejected.append({'index': index, 'reason': 'empty text', 'item': item})
            continue
        if not isinstance(when, str):
            rejected.append({'index': index, 'reason': 'time_expression not a string', 'item': item})
            continue
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            rejected.append({'index': index, 'reason': 'evidence_ids missing', 'item': item})
            continue
        if any(i not in known for i in ids):
            rejected.append({'index': index, 'reason': 'unknown source id', 'item': item})
            continue
        if any(session_no(next(t['session'] for t in conversation['turns'] if t['id'] == i)) > limit for i in ids):
            rejected.append({'index': index, 'reason': 'future source id', 'item': item})
            continue
        if not (set(ids) & current):
            rejected.append({'index': index, 'reason': 'no current-session source', 'item': item})
            continue
        facts.append({'id': 'fact_%02d' % index, 'text': text.strip(),
                      'time_expression': when.strip(),
                      'evidence_ids': list(dict.fromkeys(ids))})
    return facts, rejected

def _record_fields(archive, rid, kind):
    'Full structural view of one stored record, including its reusable id.'
    if kind == 'event':
        e = archive.events[rid]
        return {'id': rid, 'kind': kind, 'subject': e['subject'], 'action': e['action'],
                'description': e['description'], 'thread': e['thread'],
                'time_expression': e['time']['raw_expression'],
                'time_anchor': e['time']['semantic_anchor'],
                'bounds': [e['time']['start'], e['time']['end']],
                'precision': e['time']['precision'],
                'status': e.get('status', 'uncertain'),
                'evidence_ids': e['evidence_ids'], 'mentions': len(e.get('mentions', []))}
    if kind in ('current_state', 'state_history'):
        if kind == 'current_state':
            person, attr = rid.split(':')[1], rid.split(':', 2)[2]
            x = archive.states[person][attr]
        else:
            x = archive.state_history[int(rid.split(':')[1])]
        return {'id': rid, 'kind': kind, 'subject': x['subject'],
                'attribute': x['attribute'], 'value': x['value'],
                'mode': x.get('mode'), 'valid_from': x.get('valid_from'),
                'as_of': x.get('as_of'), 'temporal_scope': x.get('temporal_scope'),
                'recorded_at': x.get('recorded_at'), 'evidence_ids': x['evidence_ids']}
    if kind == 'ledger':
        _, group, key = rid.split(':', 2)
        x = archive.ledgers[group][key]
        return {'id': rid, 'kind': kind, 'group': group, 'item_key': key,
                'subject': x['subject'], 'value': x['value'],
                'recorded_at': x.get('recorded_at'), 'evidence_ids': x['evidence_ids']}
    if kind == 'detail':
        x = archive.details[rid]
        return {'id': rid, 'kind': kind, 'subject': x['subject'],
                'description': x['description'], 'entity': x.get('entity', ''),
                'evidence_ids': x['evidence_ids']}
    if kind == 'loop':
        x = archive.loops[rid]
        return {'id': rid, 'kind': kind, 'initiator': x['initiator'],
                'recipient': x['recipient'], 'description': x['description'],
                'type': x.get('type'), 'status': x.get('status'),
                'resolution': x.get('resolution', ''),
                'evidence_ids': x['evidence_ids']}
    raise KeyError(rid)

def render_record(fields):
    head = '[%s | %s]' % (fields['kind'], fields['id'])
    if fields['kind'] == 'event':
        body = ('subject: %s | action: %s | thread: %s\n  description: %s\n'
                '  time: %r -> %s..%s (%s)\n  status: %s | mentions: %d'
                % (fields['subject'], fields['action'], fields['thread'],
                   fields['description'], fields['time_expression'],
                   fields['bounds'][0], fields['bounds'][1], fields['precision'],
                   fields['status'], fields['mentions']))
    elif fields['kind'] in ('current_state', 'state_history'):
        body = ('subject: %s | attribute: %s | value: %s | mode: %s\n'
                '  valid_from: %s | as_of: %s | scope: %s | reported: %s'
                % (fields['subject'], fields['attribute'], fields['value'],
                   fields['mode'], fields['valid_from'], fields['as_of'],
                   fields['temporal_scope'], fields['recorded_at']))
    elif fields['kind'] == 'ledger':
        body = ('subject: %s | group: %s | item_key: %s | value: %s | reported: %s'
                % (fields['subject'], fields['group'], fields['item_key'],
                   fields['value'], fields['recorded_at']))
    elif fields['kind'] == 'detail':
        body = ('subject: %s | entity: %s\n  description: %s'
                % (fields['subject'], fields['entity'], fields['description']))
    else:
        body = ('%s -> %s | type: %s | status: %s\n  description: %s\n  resolution: %s'
                % (fields['initiator'], fields['recipient'], fields['type'],
                   fields['status'], fields['description'], fields['resolution']))
    return head + ' ' + body + '\n  sources: ' + ', '.join(fields['evidence_ids'])


class RecallIndex:
    """Recall relevant records already written by the hierarchical compiler."""
    def __init__(self, archive, embed_client=None):
        self.archive = archive
        self.embed_client = embed_client
        self.vectors = {}
        self.refresh()

    def refresh(self):
        self.records = memory_records(self.archive)
        self.by_id = {record['id']: record for record in self.records}
        texts = [record['text'] for record in self.records]
        self.tokens = [text.lower().split() or ['_empty'] for text in texts]
        self.bm = BM25Okapi(self.tokens) if texts else None
        self.fields = {record['id']: _record_fields(self.archive, record['id'], record['kind'])
                       for record in self.records}
        if self.embed_client is not None:
            missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
            self.vectors.update(zip(missing, self.embed_client.embed(missing, 'memory_index')))

    def _dense_scores(self, query):
        if self.embed_client is None or not self.records:
            return None
        query_vector = np.asarray(self.embed_client.embed([query], 'fact_query')[0], dtype=float)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = []
        for record in self.records:
            vector = np.asarray(self.vectors[record['text']], dtype=float)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            scores.append(float(vector @ query_vector))
        return np.asarray(scores)

    def for_fact(self, fact, limit=PER_FACT_LIMIT):
        'One ranked candidate list per fact, fused from lexical and vector ranks.'
        if not self.records:
            return []
        query = fact['text']
        if fact.get('time_expression'):
            query = query + ' ' + fact['time_expression']
        rrf = {}
        lexical = self.bm.get_scores(query.lower().split())
        for rank, i in enumerate(np.argsort(-lexical)[:60]):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
        dense = self._dense_scores(query)
        if dense is not None:
            for rank, i in enumerate(np.argsort(-dense)[:60]):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
        low = query.casefold()
        scored = []
        for i, score in rrf.items():
            r = self.records[i]
    
    
            subject = str(r.get('subject') or '').casefold()
            if subject and subject in low:
                score += 0.05
            when = (self.fields.get(r['id']) or {}).get('recorded_at') or ''
            scored.append((round(score, 6), when, r['id']))
    
        scored.sort(key=lambda x: (-x[0], _desc(x[1]), x[2]))
        return [rid for _, _, rid in scored[:limit]]

def _desc(text):
    return ''.join(chr(0x10FFFF - ord(c)) for c in str(text))

def organize_prompt(prompts, facts, memory_blocks, source_lines):
    fact_view = {'facts': [{k: f[k] for k in ('id', 'text', 'time_expression', 'evidence_ids')}
                           for f in facts]}
    text = prompts['organize']['text'].rstrip('\n')
    return (text + '\n\nFACTS:\n' + json.dumps(fact_view, ensure_ascii=False, indent=1)
            + '\n\nEXISTING_MEMORY:\n' + ('\n\n'.join(memory_blocks) or '(empty)')
            + '\n\nSOURCES:\n' + ('\n'.join(source_lines) or '(none)'))

class SourceBudgetError(RuntimeError):
    "The fact's own passages do not fit; cutting them would hide the check."

def _line_cost(text):
    return tokens(text + '\n')

def collect(facts, index, archive, conversation, session,
            memory_budget=MEMORY_BUDGET, source_budget=SOURCE_BUDGET):
    "Per-fact recall where the fact's own passages are reserved first."
    visible = {t['id']: t for t in visible_turns(conversation, session['id'])}
    mapping, merged = {}, {}
    for fact in facts:
        ids = index.for_fact(fact)
        mapping[fact['id']] = ids
        for rid in ids:
            merged.setdefault(rid, []).append(fact['id'])
    order = sorted(merged, key=lambda rid: (-len(merged[rid]), rid))

    source_lines, shown_source_ids = [], []
    fact_source_ids, unverified = [], []
    used_sources = 0
    for fact in facts:
        for sid in fact['evidence_ids']:
            if sid in fact_source_ids:
                continue
            if sid not in visible:
                unverified.append(sid)
                continue
            line = render_turn(visible[sid])
            cost = _line_cost(line)
            if used_sources + cost > source_budget:
                raise SourceBudgetError(
                    'fact sources need more than %d tokens at %s'
                    % (source_budget, session['id']))
            fact_source_ids.append(sid)
            source_lines.append(line)
            shown_source_ids.append(sid)
            used_sources += cost

    blocks, kept_ids, cut_ids, extra_pool = [], [], [], []
    used_memory = 0
    for rid in order:
        fields = index.fields.get(rid)
        if not fields:
            cut_ids.append(rid)
            continue
        block = render_record(fields)
        cost = _line_cost(block)
        extra = [sid for sid in fields['evidence_ids']
                 if sid in visible and sid not in shown_source_ids]
        extra_lines = [render_turn(visible[sid]) for sid in extra]
        extra_cost = sum(_line_cost(line) for line in extra_lines)
        for sid in extra:
            if sid not in extra_pool:
                extra_pool.append(sid)
        if used_memory + cost > memory_budget or used_sources + extra_cost > source_budget:

            cut_ids.append(rid)
            continue
        blocks.append(block)
        kept_ids.append(rid)
        used_memory += cost
        for sid, line in zip(extra, extra_lines):
            if sid in shown_source_ids:
                continue
            source_lines.append(line)
            shown_source_ids.append(sid)
            used_sources += _line_cost(line)

    cut_source_ids = [sid for sid in extra_pool if sid not in shown_source_ids]
    return {
        'mapping': mapping, 'memory_blocks': blocks,
        'kept_ids': sorted(kept_ids), 'cut_ids': sorted(cut_ids),
        'sources': source_lines, 'shown_source_ids': shown_source_ids,
        'cut_source_ids': cut_source_ids, 'fact_source_ids': fact_source_ids,
        'unverified_source_ids': unverified,
        'memory_used': used_memory, 'source_used': used_sources,
        'memory_budget': memory_budget, 'source_budget': source_budget,
        'memory_cut': len(cut_ids), 'source_cut': len(cut_source_ids),
    }

def filter_delta(delta, shown_source_ids):
    'Only sources the second stage actually received can support a record.'
    allowed = set(shown_source_ids or ())
    out = {kind: [] for kind in KINDS}
    rejected, stripped = [], []
    for kind in KINDS:
        for item in delta.get(kind) or []:
            ids = list(item.get('evidence_ids') or [])
            kept = [i for i in ids if i in allowed]
            dropped = [i for i in ids if i not in allowed]
            if not kept:
                rejected.append({'kind': kind, 'item': item, 'reason': 'unshown source'})
                continue
            if dropped:
                item = dict(item)
                item['evidence_ids'] = kept
                stripped.append({'kind': kind, 'kept': kept, 'dropped': dropped})
            out[kind].append(item)
    return out, rejected, stripped


def build_two_stage(conversation, client, embed_client=None, prompts=None,
                    memory_budget=MEMORY_BUDGET, source_budget=SOURCE_BUDGET,
                    fact_tokens=3000, organize_tokens=12000):
    """Extract facts, recall earlier memory, and apply evidence-linked updates."""
    prompts = prompts or load_prompts()
    archive = Archive(conversation)
    index = RecallIndex(archive, embed_client)
    fact_sessions = []
    for session in conversation['sessions']:
        prompt = facts_prompt(prompts, session_text(conversation, session),
                              reference_context(conversation, session))
        payload = parse_json(client.chat(prompt, 'facts', max_tokens=fact_tokens))
        facts, _ = validate_facts(payload, conversation, session)
        fact_sessions.append({'session': session['id'], 'facts': facts})
        if facts:
            recall = collect(facts, index, archive, conversation, session, memory_budget, source_budget)
            prompt = organize_prompt(prompts, facts, recall['memory_blocks'], recall['sources'])
            delta = parse_json(client.chat(prompt, 'organize', max_tokens=organize_tokens))
            delta, _, _ = filter_delta(delta, recall['shown_source_ids'])
            archive.apply(delta, session['id'])
            index.refresh()
    return archive, fact_sessions
