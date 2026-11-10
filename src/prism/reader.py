"""Read original evidence and apply the default answer-normalization path."""
import re

from .io import parse_json
from .prompts import ANSWER, INFER, DURATION
from .normalization import (present_answer, canonical_count, needs_unit, canonical_verdict,
                            strip_leading_hedge, drop_inline_hedges, strip_leading_subject,
                            strip_note_prefix, strip_trailer, strip_as_of)

ABSTAINED = __import__('re').compile(
    r'^\s*(not established|not clearly established|insufficient|unknown|not mentioned|'
    r'no information|unclear|cannot determine|not explicitly stated)', 2)


def infer_answer(question, evidence, client):
    prompt = INFER + '\nFINDINGS:\n' + evidence + '\nQUESTION:\n' + question
    return parse_json(client.chat(prompt, max_tokens=1000))


def read_answer(question, trace, workspace, client):
    evidence = trace['context']['text']
    prompt = ANSWER + '\nEVIDENCE:\n' + evidence + '\nQUESTION:\n' + question
    answer = parse_json(client.chat(prompt, max_tokens=1000))
    value = answer['answer']
    if isinstance(value, list):
        value = ', '.join(str(item).strip() for item in value if str(item).strip())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('empty answer')

    # The frozen reader consults its research synthesis when its first answer abstains.
    if re.match(r'\s*(not established|not clearly established|insufficient|unknown|not mentioned|'
                r'no information|unclear|cannot determine|not explicitly stated)', value, re.I):
        notes = str(trace['last_state'].get('synthesis') or '')
        if notes.strip():
            prompt = ('Answer the question with the date or time the notes support.\n'
                      'QUESTION: ' + question + '\nNOTES:\n' + notes[:3000] +
                      '\nReturn JSON {"answer": string}')
            candidate = str(parse_json(client.chat(prompt, max_tokens=1000)).get('answer') or '').strip()
            if candidate and not re.match(r'\s*(not established|unknown|unclear|insufficient|not mentioned)', candidate, re.I):
                value = candidate
        if ABSTAINED.match(value.strip()):
            candidate = str(infer_answer(question, evidence, client).get('answer') or '').strip()
            if candidate and not ABSTAINED.match(candidate):
                value = candidate

    anchor = next((workspace.turns[source_id].get('recorded_at')
                   for source_id in answer.get('evidence_ids', [])
                   if source_id in workspace.turns and workspace.turns[source_id].get('recorded_at')), None)
    value = canonical_count(question, present_answer(value, anchor))
    if ABSTAINED.match(value.strip()):
        candidate = str(infer_answer(question, evidence, client).get('answer') or '').strip()
        if candidate and not ABSTAINED.match(candidate):
            value = present_answer(candidate, anchor)
            answer['status'] = 'answered'
    if needs_unit(question, value):
        prompt = DURATION + '\nFINDINGS:\n' + evidence + '\nQUESTION:\n' + question
        candidate = str(parse_json(client.chat(prompt, max_tokens=1000)).get('answer') or '').strip()
        if candidate and not needs_unit(question, candidate):
            value = present_answer(candidate, anchor)

    value = canonical_verdict(question, value) or value
    value = strip_leading_hedge(value) or value
    value = drop_inline_hedges(value) or value
    value = strip_leading_subject(value, workspace.actor.get('participants') or []) or value
    for normalize in (strip_note_prefix, strip_trailer, strip_as_of):
        value = normalize(value) or value
    return {**answer, 'answer': value}
