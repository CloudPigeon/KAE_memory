'Explicit allowlist between benchmark labels and online memory.'
import re

def sanitize_conversation(sample):
    cid=sample['sample_id']; c=sample['conversation']
    people=[c['speaker_a'],c['speaker_b']];sessions=[];turns=[]
    for key in sorted((k for k in c if re.fullmatch(r'session_\d+',k)),key=lambda k:int(k.split('_')[1])):
        recorded=c[key+'_date_time']; ids=[]
        for t in c[key]:
            sid=f"{cid}:{t['dia_id']}"; ids.append(sid)
            turns.append({'id':sid,'dia_id':t['dia_id'],'speaker':t['speaker'],
                'listener':next((x for x in people if x!=t['speaker']),''),'text':t['text'],
                'caption':t.get('blip_caption',''),'image_query':t.get('query',''),
                'session':key,'recorded_at':recorded})
        sessions.append({'id':key,'recorded_at':recorded,'source_ids':ids})
    return {'id':cid,'participants':people,'sessions':sessions,'turns':turns}

def render_turn(t):
    out=f"[{t['id']}] [Speaker: {t['speaker']} | Listener: {t['listener']}] [Recorded: {t['recorded_at']}] {t['text']}"
    if t.get('caption'): out+=f"\n[Visual description, possibly imperfect]: {t['caption']}"
    if t.get('image_query'): out+=f"\n[Image search query, not a verified visual fact]: {t['image_query']}"
    return out
