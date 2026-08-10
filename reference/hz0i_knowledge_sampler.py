"""Domain-balanced knowledge-dense batch sampler for HZ-0I."""
from __future__ import annotations
import json,random
from pathlib import Path
class KnowledgeDenseSampler:
 def __init__(self,paths,weights=None,seed=0,deduplicate=True):
  self.rng=random.Random(seed);self.domains={}
  for k,v in paths.items():
   rows=[json.loads(x) for x in Path(v).open() if x.strip()]
   if deduplicate:
    seen=set();unique=[]
    for row in rows:
     key=tuple(row) if isinstance(row,list) else json.dumps(row,sort_keys=True)
     if key not in seen: seen.add(key);unique.append(row)
    rows=unique
   if not rows: raise ValueError(f"domain {k} has no rows")
   self.domains[k]=rows;self.names=list(self.domains);self.weights=weights or {k:1/len(self.names) for k in self.names}
 def sample_stratified(self,batch_size,seq_len):
  names=self.names*((batch_size+len(self.names)-1)//len(self.names));self.rng.shuffle(names);names=names[:batch_size];return [(k,[int(x) for x in self.rng.choice(self.domains[k])[:seq_len]]) for k in names]
 def sample(self,batch_size,seq_len):
  names=self.rng.choices(self.names,weights=[self.weights[k] for k in self.names],k=batch_size); rows=[]
  for k in names:
   row=self.rng.choice(self.domains[k]); vals=[int(x) for x in row[:seq_len]]; rows.append((k,vals))
  return rows
 def distribution(self,n):
  out={k:0 for k in self.names}
  for _ in range(n): out[self.rng.choices(self.names,weights=[self.weights[k] for k in self.names])[0]]+=1
  return out


class AdaptiveKnowledgeSampler(KnowledgeDenseSampler):
 def __init__(self,paths,weights=None,seed=0,temperature=1.0,min_weight=0.05):
  super().__init__(paths,weights,seed);self.temperature=temperature;self.min_weight=min_weight;self.loss_ema={k:1.0 for k in self.names}
 def update_losses(self,losses,decay=0.9):
  for k,v in losses.items():
   if k in self.loss_ema:self.loss_ema[k]=decay*self.loss_ema[k]+(1-decay)*float(v)
  raw={k:self.loss_ema[k]**self.temperature for k in self.names};z=sum(raw.values());w={k:v/z for k,v in raw.items()};floor=min(self.min_weight,1.0/len(self.names));remaining=1-floor*len(self.names);self.weights={k:floor+remaining*v for k,v in w.items()}


# Lightweight checkpoint protocol for continual-training resumes.
def _sampler_state(self):
 return {'weights':dict(self.weights),'rng_state':self.rng.getstate(),'names':list(self.names)}
def _sampler_load(self,state):
 saved_names=state.get('names',self.names)
 if set(saved_names)!=set(self.names): raise ValueError(f'sampler checkpoint domains {saved_names} do not match current domains {self.names}')
 self.weights=dict(state.get('weights',self.weights));self.rng.setstate(state['rng_state'])
KnowledgeDenseSampler.state_dict=_sampler_state
KnowledgeDenseSampler.load_state_dict=_sampler_load
def _adaptive_state(self):
 return {'weights':dict(self.weights),'rng_state':self.rng.getstate(),'names':list(self.names),'loss_ema':dict(self.loss_ema),'min_weight':self.min_weight,'temperature':self.temperature}
def _adaptive_load(self,state):
 _sampler_load(self,state);self.loss_ema=dict(state.get('loss_ema',self.loss_ema));self.min_weight=float(state.get('min_weight',self.min_weight));self.temperature=float(state.get('temperature',self.temperature))
AdaptiveKnowledgeSampler.state_dict=_adaptive_state
AdaptiveKnowledgeSampler.load_state_dict=_adaptive_load
