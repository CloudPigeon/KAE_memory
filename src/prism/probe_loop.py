"""Six-view turn ranking and the research agent's evidence tools."""
import re
import numpy as np
from rank_bm25 import BM25Okapi

from .compilation import clean_tokenize_algorithmic
from .data import render_turn
from .retrieve import rrf_order, tokens, unit_vectors


def back_project(turn_scores, record_scores, links, second_weight):
    """Project each cited memory's relevance onto its original turns."""
    projected = turn_scores.copy()
    for turn, indices in links.items():
        if indices:
            scores = sorted((float(record_scores[i]) for i in indices), reverse=True)
            score = scores[0] + (second_weight * scores[1] if len(scores) > 1 else 0.0)
            projected[int(turn)] = max(projected[int(turn)], score)
    return projected


class Workspace:
    def __init__(self, actor, memory, raw, client, multiview, context_chars=18000, search_top_k=5):
        self.actor, self.memory, self.raw, self.client = actor, memory, raw, client
        self.turns = {turn['id']: turn for turn in actor['turns']}
        self.positions = {turn['id']: i for i, turn in enumerate(actor['turns'])}
        self.bank, self.seen_memories, self.pages, self.history = {}, {}, {}, []
        self.context_chars, self.search_top_k = context_chars, search_top_k
        self.tids = multiview['tids']
        self.compiled = unit_vectors(multiview['clean_vecs'])
        self.organized = unit_vectors(multiview['org_vecs'])
        self.fact_links = multiview['turn_fact_idxs']
        self.schema_links = multiview['turn_org_idxs']
        self.lexical = BM25Okapi(multiview['bm25_raw_tokens'])
        self.lexical_stem = BM25Okapi(multiview['bm25_stem_tokens'])

    def rank_turns(self, query, vector=None):
        """Fuse raw, compiled, fact, schema and two lexical rankings."""
        lexical = self.lexical.get_scores(re.findall(r'\w+', query.lower()) or ['_empty'])
        lexical_stem = self.lexical_stem.get_scores(clean_tokenize_algorithmic(query))
        rankings = [np.argsort(-lexical), np.argsort(-lexical_stem)]
        if vector is not None:
            query_vector = np.asarray(vector, dtype=float)
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            raw = self.raw.v @ query_vector
            compiled = self.compiled @ query_vector
            facts = np.maximum(compiled, raw)
            if self.memory.v.size:
                facts = back_project(facts, self.memory.v @ query_vector, self.fact_links, 0.12)
            schemas = facts.copy()
            if self.organized.size:
                schemas = back_project(schemas, self.organized @ query_vector, self.schema_links, 0.10)
            rankings = [np.argsort(-scores) for scores in (raw, compiled, facts, schemas)] + rankings
        return [self.tids[i] for i in rrf_order(rankings)]

    def neighbors(self, ids, radius=1):
        found = set()
        for source_id in ids:
            if source_id in self.turns:
                position = self.positions[source_id]
                session = self.turns[source_id]['session']
                found.update(turn['id'] for turn in self.actor['turns'][max(0, position-radius):position+radius+1]
                             if turn['session'] == session)
        return sorted(found, key=self.positions.get)

    def originals(self, ids):
        return '\n'.join(render_turn(self.turns[source_id]) for source_id in ids if source_id in self.turns)

    def execute(self, action):
        if action.get('tool') == 'search':
            action = {**action, 'limit': min(self.search_top_k, max(1, int(action.get('limit', self.search_top_k))))}
        tool = action.get('tool')
        hits, ids, meta = [], [], {'action': action}
        if tool == 'search':
            query = str(action.get('query', '')).strip()
            if not query:
                return {**meta, 'error': 'Empty query'}
            mode = 'keyword' if action.get('mode') == 'keyword' else 'hybrid'
            channel, size = action.get('channel', 'both'), action['limit']
            page = (' '.join(query.lower().split()), mode, channel)
            offset = self.pages.get(page, 0)
            self.pages[page] = offset + size
            vector = self.client.embed([query], 'probe_query_embedding')[0] if mode == 'hybrid' else None
            candidates = {}
            for name, index in [('memory', self.memory), ('raw', self.raw)]:
                if channel in ('both', name):
                    ranked = index.recall(query, vector, mode=mode)
                    candidates[name] = {'returned': len(ranked[offset:offset+size]), 'total': len(ranked)}
            ranked_turns = self.rank_turns(query, vector)
            start, count = (offset, size) if channel in ('raw', 'memory') else (offset * 2, size * 2)
            anchors = ranked_turns[start:start+count]
            hits = [self.raw.records[self.positions[source_id]] for source_id in anchors]
            ids = self.neighbors(anchors, 1)
            meta.update(offset=offset, limit=size, candidates=candidates, anchors_selected=anchors)
        elif tool == 'read':
            known = [source_id for source_id in action.get('source_ids', []) if source_id in self.bank]
            ids = self.neighbors(known, max(1, min(8, int(action.get('radius', 2)))))
            if not known:
                meta['error'] = 'Use source IDs returned by an earlier tool.'
        elif tool == 'thread':
            seed = self.seen_memories.get(action.get('memory_id'))
            if seed:
                hits = [record for record in self.memory.records if record['id'] == seed['id'] or
                        (seed.get('group') and record.get('group') == seed['group'] and record['subject'] == seed['subject'])]
                ids = self.neighbors([source_id for record in hits for source_id in record['evidence_ids']], 1)
            else:
                meta['error'] = 'Use a memory ID returned by an earlier tool.'
        elif tool == 'session':
            session = action.get('session_id')
            if any(turn['session'] == session for turn in self.bank.values()):
                ids = [turn['id'] for turn in self.actor['turns'] if turn['session'] == session]
            else:
                meta['error'] = 'Use a session encountered in earlier evidence.'
        else:
            meta['error'] = 'Unknown tool; use search, read, thread or session.'

        ids = [source_id for source_id in ids if source_id in self.turns]
        new = [source_id for source_id in ids if source_id not in self.bank]
        self.bank.update({source_id: self.turns[source_id] for source_id in ids})
        memories = [record for record in hits if record['kind'] != 'raw']
        self.seen_memories.update({record['id']: record for record in memories})
        self.history.append({**meta, 'new_source_count': len(new), 'returned_source_count': len(ids)})
        return {**meta, 'memory_clues': memories, 'source_ids': ids,
                'new_source_ids': new, 'originals': self.originals(ids),
                'session_ids': sorted({self.turns[source_id]['session'] for source_id in ids})}

    def supporting_ids(self, state):
        return list(dict.fromkeys(source_id for need in state.get('needs', [])
                                  for source_id in need.get('source_ids', []) if source_id in self.bank))

    def final_context(self, state):
        """Pin cited neighborhoods, then fill the reader budget with encountered turns."""
        pinned = self.neighbors(self.supporting_ids(state), 2)
        ordered = list(dict.fromkeys(pinned + list(self.bank)))
        kept, used = [], 0
        for source_id in ordered:
            if source_id in self.bank:
                size = len(self.bank[source_id]['text'])
                if used + size > self.context_chars and kept:
                    break
                kept.append(source_id)
                used += size
        text = 'Original conversation evidence:\n' + self.originals(kept)
        return {'text': text, 'source_ids': kept, 'memory_ids': [],
                'units': tokens(text), 'fair_flat': True, 'fifo': False}
