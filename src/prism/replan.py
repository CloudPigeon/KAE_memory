"""Replan unresolved needs using known original sources."""
import json
from .io import parse_json
from .prompts import REPLAN, TOOLS

def repair_plan(question, state, workspace, verification, step, call):
    ids=workspace.neighbors(workspace.supporting_ids(state),2)
    data={'original_question':question,'notebook':state,'verification':verification,
          'supporting_originals':workspace.originals(ids),'tool_history':workspace.history}
    value=parse_json(call(REPLAN+'\n'+TOOLS+'\nINPUT DATA\n'+json.dumps(data,ensure_ascii=False),
                          'replan',step,4000))

    exhausted=[h['action'] for h in workspace.history if h.get('new_source_count')==0 and
               h['action'].get('tool') in ('read','thread','session')]
    value['next_actions']=[a for a in value.get('next_actions',[]) if a not in exhausted]
    return value
