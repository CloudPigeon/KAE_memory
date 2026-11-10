'Evidence-linked temporal memory archive.'
import copy
import hashlib
import json
from .time import anchor,date_of,is_ordinal

def _axis(value):
    'Which ordering axis a timestamp belongs to (observation vs calendar).'
    if value is None:
        return 'none'
    if isinstance(value, str) and value.startswith('ordinal:'):
        return 'ordinal'
    return 'calendar'


def _order_key(value):
    'Observation order compares numerically; calendar dates by date. Never as strings.'
    if value is None:
        return (0, 0)
    if isinstance(value, str) and value.startswith('ordinal:'):
        return (1, int(value.split(':', 1)[1]))
    return (2, date_of(value).toordinal())


def uid(prefix,value):
    return prefix+'_'+hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]

class Archive:
    def __init__(self, conversation):
        self.conversation=conversation
        self.sources={t['id']:t for t in conversation['turns']}
        self.events={};self.states={};self.state_history=[];self.ledgers={}
        self.details={};self.loops={};self.threads={};self.rejected=[];self.applied=[]

    def reject(self,kind,item,reason):
        self.rejected.append({'kind':kind,'item':item,'reason':reason})

    def valid(self,item,session):
        ids=item.get('evidence_ids')
        if not isinstance(ids,list) or not ids or any(i not in self.sources for i in ids):return False

        n=int(session.split('_')[1])
        return all(int(self.sources[i]['session'].split('_')[1])<=n for i in ids) and any(self.sources[i]['session']==session for i in ids)

    def apply(self,delta,session):
        if session in self.applied:return
        stamp=next(s['recorded_at'] for s in self.conversation['sessions'] if s['id']==session)
        for kind in ('events','updates','details','loops'):
            values=delta.get(kind,[])
            if not isinstance(values,list):raise ValueError(f'{kind} must be an array')
            for original in values:
                item=copy.deepcopy(original)
                if not isinstance(item,dict) or not self.valid(item,session):
                    self.reject(kind,item,'invalid, future or non-current evidence');continue
                item['evidence_ids']=list(dict.fromkeys(item['evidence_ids']))
                item['recorded_at']=stamp
                try:getattr(self,'_'+kind)(item)
                except (KeyError,TypeError,ValueError) as e:self.reject(kind,item,str(e))
        self.applied.append(session)

    def _events(self,x):
        if not all(isinstance(x.get(k),str) and x[k].strip() for k in ('subject','action','description','thread')):
            raise ValueError('event fields missing')


        role='plan' if str(x.get('status','')).strip().lower()=='planned' else 'event'
        if is_ordinal(x['recorded_at']):

            x['time']={'raw_expression':x.get('time_expression',''),
                       'recorded_at':x['recorded_at'],
                       'semantic_anchor':'unknown (observation order only)',
                       'start':None,'end':None,'precision':'unknown'}
        else:
            x['time']=anchor(x.get('time_expression',''),x['recorded_at'],role=role)
        oldid=x.pop('existing_event_id','')
        if oldid:
            if oldid not in self.events:raise ValueError('unknown event identity')
            old=self.events[oldid]


            if old['subject'].casefold()!=x['subject'].casefold():
                self.reject('event_link',copy.deepcopy(x),'cross-subject link kept as separate unresolved event')
                x['unresolved_existing_event_id']=oldid
                return self._events(x)
            if old['action'].casefold()!=x['action'].casefold():
                self.reject('event_link',copy.deepcopy(x),'cross-action link kept as separate unresolved event')
                x['unresolved_existing_event_id']=oldid
                return self._events(x)
            a,b=old['time'],x['time']
            if a['start'] and b['start'] and (a['end']<b['start'] or b['end']<a['start']):
                self.reject('event_link',copy.deepcopy(x),'disjoint date link kept as separate unresolved event')
                x['unresolved_existing_event_id']=oldid
                return self._events(x)
            old['mentions'].append(x)
            old['evidence_ids']=list(dict.fromkeys(old['evidence_ids']+x['evidence_ids']))
            if date_of(x['recorded_at'])>=date_of(old['recorded_at']):
                old['description']=x['description'];old['status']=x.get('status',old.get('status','uncertain'))
                old['recorded_at']=x['recorded_at']

            if not a['start'] and b['start']:old['time']=b
            return
        eid=uid('evt',[x['subject'],x['action'],x['description'],x['evidence_ids']])
        self.events[eid]={**x,'id':eid,'mentions':[copy.deepcopy(x)]}
        key=x['subject']+'|'+x['thread']
        th=self.threads.setdefault(key,{'subject':x['subject'],'thread':x['thread'],'event_ids':[]})
        if eid not in th['event_ids']:th['event_ids'].append(eid)

    def _updates(self,x):
        person,attr,mode=x['subject'],x['attribute'],x['mode']
        if not isinstance(x.get('value'),str) or not x['value']:raise ValueError('empty update value')
        effective=x.get('valid_from')
        x['valid_from']=date_of(effective).isoformat() if effective else None
        if mode=='SET_SINGLE_VALUE':
            scope=x.get('temporal_scope','unspecified')
            if scope not in ('current','historical','unspecified'):raise ValueError('invalid temporal scope')


            x['as_of']=date_of(x['recorded_at']).isoformat() if scope=='current' else x['valid_from']
            self.state_history.append(x)
            if scope != 'current':


                return
            d=self.states.setdefault(person,{})
            old=d.get(attr,{})
            prior=old.get('as_of')
            if attr not in d or prior is None:
                d[attr]=x
            elif _axis(x['as_of'])==_axis(prior):
                if _order_key(x['as_of']) >= _order_key(prior):
                    d[attr]=x

        elif mode=='APPEND_TO_LEDGER':

            identity=x.get('item_key') or x['value'].casefold()
            group=self.ledgers.setdefault(person+'|'+attr,{})
            if identity in group:
                old=group[identity];history=old.setdefault('versions',[])
                history.append({k:copy.deepcopy(v) for k,v in old.items() if k!='versions'})
                old['evidence_ids']=list(dict.fromkeys(old['evidence_ids']+x['evidence_ids']))
                if date_of(x['recorded_at'])>=date_of(old['recorded_at']):
                    old['value']=x['value'];old['recorded_at']=x['recorded_at']
            else:group[identity]=x
        else:raise ValueError('unknown update mode')

    def _details(self,x):
        if not x.get('subject') or not x.get('description'):raise ValueError('detail fields missing')
        key=uid('detail',[x['subject'],x['description']])
        if key in self.details:x['evidence_ids']=list(dict.fromkeys(self.details[key]['evidence_ids']+x['evidence_ids']))
        self.details[key]={**x,'id':key}

    def _loops(self,x):
        for k in ('initiator','recipient','description','status'):
            if not isinstance(x.get(k),str):raise ValueError('loop fields missing')
        oldid=x.pop('existing_loop_id','')
        if oldid:
            if oldid not in self.loops:raise ValueError('unknown loop identity')
            old=self.loops[oldid]
            if (old['initiator'],old['recipient'])!=(x['initiator'],x['recipient']):
                self.reject('loop_link',copy.deepcopy(x),'cross-person link kept as separate unresolved loop')
                x['unresolved_existing_loop_id']=oldid
                return self._loops(x)
            old.setdefault('history',[]).append(copy.deepcopy(x))
            old['evidence_ids']=list(dict.fromkeys(old['evidence_ids']+x['evidence_ids']))


            if (x.get('resolution_confidence') or 0)>=0.9:
                old['status']=x['status'];old['resolution']=x.get('resolution','')
            else:old.setdefault('unconfirmed_resolutions',[]).append(x)
        else:
            key=uid('loop',[x['initiator'],x['recipient'],x['description'],x['evidence_ids']])
            self.loops[key]={**x,'id':key,'history':[]}

    def to_dict(self):
        return {k:copy.deepcopy(getattr(self,k)) for k in ('events','states','state_history','ledgers','details','loops','threads','rejected','applied')}

    def restore(self,value):
        for k in self.to_dict():setattr(self,k,value[k])
        return self
