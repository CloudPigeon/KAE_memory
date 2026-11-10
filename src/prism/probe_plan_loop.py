"""Probe, plan and integrate evidence in a bounded need notebook."""
import json
from .io import parse_json
from .prompts import PROBE, PLAN, CONTROL, TOOLS

def broaden_query(text):
    'Drop qualifiers that make a search too narrow to ever return evidence.'
    q = str(text).strip()
    q = __import__('re').sub(
        r'^\s*(?:the\s+)?(?:exact|specific|particular|precise)\s+'
        r'(?:date|time|time period|date or time period|city or town|city|place|name|number)?'
        r'\s*(?:of|when|that)?\s*', '', q, flags=2)
    q = __import__('re').sub(r'\b(?:or town|or time period|or period)\b', ' ', q, flags=2)
    q = __import__('re').sub(r'\b(?:exact|specific|particular|detailed|precise)\b', ' ', q, flags=2)
    q = __import__('re').sub(
        r'\b(?:is|was|were|did|does|do|the|a|an|of|to|in|on|at|for|from|by|with|about'
        r'|who|what|when|where|why|how)\b',
        ' ', q, flags=2)
    q = __import__('re').sub(r'\s+', ' ', q).strip(' ?.')
    words = q.split()
    if not words:
        return str(text).strip()
    return ' '.join(words[:12])

def update_notebook(state, decision, known_sources):
    needs = {n['need_id']: dict(n) for n in state['needs']}
    invalid = []
    for patch in decision.get('updates', []):
        nid = patch.get('need_id')
        if nid not in needs:
            invalid.append({'need_id': nid, 'error': 'unknown requirement'}); continue
        ids = patch.get('source_ids', [])
        unknown = [sid for sid in ids if sid not in known_sources]
        item = needs[nid]
        item.update({k: patch[k] for k in ['finding', 'status'] if k in patch})

        if isinstance(patch.get('requires_answer'), bool):
            item['requires_answer'] = patch['requires_answer']
        item['source_ids'] = [sid for sid in ids if sid in known_sources]
        if unknown or (item.get('status') in ('supported', 'inferred', 'contradicted') and not item['source_ids']):
            item['status'] = 'open'; invalid.append({'need_id': nid, 'unknown_sources': unknown, 'error': 'source support missing'})
        elif item.get('status') in ('supported', 'inferred', 'contradicted', 'not_required'):
            item.pop('verification_gap', None)
    for addition in decision.get('add_needs', []):
        if isinstance(addition, str):
            addition = {'need': addition}
        if addition.get('need') and not any(n['need'] == addition['need'] for n in needs.values()):
            nid = 'n' + str(len(needs)+1)
            needs[nid] = {'need_id': nid, 'need': addition['need'], 'status': 'open', 'finding': '', 'source_ids': [],
                           'requires_answer': addition.get('requires_answer', True)}
    return {**state, 'needs': list(needs.values()), 'synthesis': decision.get('synthesis', state.get('synthesis', ''))}, invalid

def research(question, workspace, call, max_steps=3, verify=None, replan=None):

    workspace.question = question
    probes = parse_json(call(PROBE + '\nQuestion: ' + question, 'probe', 0, 500))
    queries = list(dict.fromkeys(str(q).strip() for q in probes.get('queries', []) if str(q).strip()))[:3] or [question]
    batches = [workspace.execute({'tool': 'search', 'query': q, 'channel': 'both', 'mode': 'hybrid', 'limit': 4}) for q in queries]
    planning_input = {'original_question': question, 'participants': workspace.actor['participants'], 'probe_results': batches}
    plan = parse_json(call(PLAN + '\n' + TOOLS + '\nINPUT DATA\n' + json.dumps(planning_input, ensure_ascii=False), 'plan', 0, 4000))
    plan_needs = [{'need': n} if isinstance(n, str) else n for n in plan['needs']]
    state = {'scope': plan.get('scope', ''), 'needs': [
        {'need_id': 'n'+str(i+1), 'need': n['need'], 'finding': n.get('clues', ''), 'source_ids': [], 'status': 'open',
         'requires_answer': n.get('requires_answer', True)}
        for i, n in enumerate(plan_needs)], 'synthesis': ''}
    if not state['needs']:
        raise ValueError('empty information plan')
    actions = plan.get('next_actions') or [{'tool': 'search', 'query': question, 'channel': 'both', 'mode': 'hybrid'}]
    trace = {'probes': probes, 'plan': plan, 'tool_results': batches.copy(), 'steps': []}
    stop = 'step_ceiling'; feedback = []
    for step in range(max_steps):


        batches = [workspace.execute(a) for a in actions[:3]]
        trace['tool_results'].extend(batches)
        support = workspace.neighbors(workspace.supporting_ids(state), 1)
        data = {'original_question': question, 'scope': state['scope'], 'notebook': state,
                'supporting_originals': workspace.originals(support), 'new_tool_results': batches,
                'tool_history': workspace.history, 'feedback': feedback}
        decision = parse_json(call(CONTROL + '\n' + TOOLS + '\nINPUT DATA\n' + json.dumps(data, ensure_ascii=False),
                                  'integrate', step, 3500))
        state, feedback = update_notebook(state, decision, workspace.bank)
        complete = all(n.get('status') in ('supported', 'inferred', 'contradicted', 'not_required') for n in state['needs'])
        verification = None; repair = None
        if verify is not None and (decision.get('ready') is True or complete):
            state, verification = verify(question, state, workspace, step)
            if not verification['ready']:
                decision['ready'] = False
                feedback.append({'verification': verification, 'instruction':
                    'Evidence check requests more work. Revisit the open requirements, repair unsupported findings, and choose targeted evidence actions.'})
                if replan is not None:
                    repair = replan(question, state, workspace, verification, step)
                    state, repair_errors = update_notebook(state, repair, workspace.bank)
                    feedback.extend(repair_errors)
                    decision['next_actions'] = repair.get('next_actions') or []
                elif not decision.get('next_actions'):
                    decision['next_actions'] = [{'tool': 'read', 'source_ids': workspace.supporting_ids(state), 'radius': 2}]
            complete = complete and verification['ready']
        trace['steps'].append({'step': step, 'decision': decision, 'notebook': state,
                               'feedback': feedback, 'verification': verification, 'repair': repair,
                               'accumulated_sources': len(workspace.bank)})
        trace['last_state'] = state


        if decision.get('ready') is True and complete:
            stop = 'ready'
            break
        actions = decision.get('next_actions') or []
        if not actions:
            if not decision.get('ready'):
                stop = 'unresolved'; break
            feedback.append({'error': 'Some original requirements remain open; complete the notebook or report the unresolved gap.'})
            actions = [{'tool': 'read', 'source_ids': workspace.supporting_ids(state), 'radius': 2}]
    trace.update(stop_reason=stop, context=workspace.final_context(state), tool_history=workspace.history)
    return trace
