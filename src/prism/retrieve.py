"""Memory records and reciprocal-rank hybrid retrieval."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from .data import render_turn
from .io import setup

def tokens(text):
    import tiktoken
    setup()
    return len(tiktoken.get_encoding('o200k_base').encode(text))

def memory_records(archive):
    rows=[]
    def add(key,kind,subject,text,ids,group=''):
        rows.append({'id':key,'kind':kind,'subject':subject,'text':text,'evidence_ids':ids,'group':group})
    for key,e in archive.events.items():
        add(key,'event',e['subject'],f"{e['subject']} | {e['action']} | {e['description']} | {e['time']['semantic_anchor']} | bounds={e['time']['start']}..{e['time']['end']} | {e.get('status','uncertain')}",e['evidence_ids'],e['subject']+'|'+e['thread'])
    for person,attrs in (archive.states.items()):
        for attr,x in attrs.items():


            add(f'state:{person}:{attr}','current_state',person,
                f"{person} {attr}: {x['value']} (current; valid from {x['valid_from'] or 'unstated'}; reported {x['recorded_at']})",
                x['evidence_ids'],person+'|'+attr)
    for i,x in (enumerate(archive.state_history)):


        add(f'history:{i}','state_history',x['subject'],
            f"{x['subject']} {x['attribute']}: {x['value']} (historical version; valid from {x['valid_from'] or 'unstated'}; "
            f"observed as of {x.get('as_of') or 'unstated'}; reported {x['recorded_at']})",
            x['evidence_ids'],x['subject']+'|'+x['attribute'])
    for group,values in archive.ledgers.items():
        for key,x in values.items():add('ledger:'+group+':'+key,'ledger',x['subject'],f"{group}: {x['value']} | reported {x['recorded_at']}",x['evidence_ids'],group)
    for key,x in archive.details.items():add(key,'detail',x['subject'],x['subject']+': '+x['description'],x['evidence_ids'])
    for key,x in archive.loops.items():
        add(key,'loop',x['initiator'],f"{x['initiator']} -> {x['recipient']}: {x['description']} | {x['status']} | resolution: {x.get('resolution','')}",x['evidence_ids'])
    return rows

def raw_records(conversation):
    return [{'id':t['id'],'kind':'raw','subject':t['speaker'],'text':render_turn(t),'evidence_ids':[t['id']],'group':''} for t in conversation['turns']]


def unit_vectors(vectors):
    values = np.asarray(vectors, dtype=float)
    if not values.size:
        return np.empty((0, 0))
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def rrf_order(rankings):
    scores = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking):
            position = int(position)
            scores[position] = scores.get(position, 0.0) + 1.0 / (61 + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Index:
    def __init__(self, records, vectors):
        self.records = records
        self.v = unit_vectors(vectors)
        texts = [record.get('index_text') or record['text'] for record in records]
        self.bm = BM25Okapi([re.findall(r'\w+', text.lower()) or ['_empty']
                            for text in texts]) if texts else None

    def recall(self, query, vector=None, mode='hybrid'):
        if not self.records:
            return []
        lexical = self.bm.get_scores(re.findall(r'\w+', query.lower()))
        if mode == 'keyword':
            order = [int(i) for i in np.argsort(-lexical) if lexical[i] > 0]
        else:
            query_vector = np.asarray(vector, dtype=float)
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            order = rrf_order([np.argsort(-(self.v @ query_vector)), np.argsort(-lexical)])
        return [self.records[i] for i in order]
