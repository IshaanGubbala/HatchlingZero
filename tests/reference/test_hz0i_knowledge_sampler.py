import json
from reference.hz0i_knowledge_sampler import KnowledgeDenseSampler
def test_sampler_is_reproducible_and_balances_domains(tmp_path):
 paths={}
 for n in ['general','code','math','json']:
  p=tmp_path/(n+'.jsonl');p.write_text(''.join(json.dumps(list(range(8)))+'\n' for _ in range(3)));paths[n]=p
 a=KnowledgeDenseSampler(paths,weights={'general':.25,'code':.25,'math':.25,'json':.25},seed=4);b=KnowledgeDenseSampler(paths,weights={'general':.25,'code':.25,'math':.25,'json':.25},seed=4);assert a.sample(10,8)==b.sample(10,8);assert sum(a.distribution(1000).values())==1000


def test_adaptive_sampler_upweights_hard_domains(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 paths={}
 for n in ['a','b']:
  q=tmp_path/(n+'.jsonl');q.write_text('[1,2,3]\n');paths[n]=q
 s=AdaptiveKnowledgeSampler(paths,weights={'a':.5,'b':.5},seed=1);s.update_losses({'a':1.,'b':5.},decay=0.0);assert s.weights['b']>s.weights['a']


def test_adaptive_sampler_checkpoint_preserves_policy(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 q=tmp_path/'a.jsonl';q.write_text('[1,2,3]\n');s=AdaptiveKnowledgeSampler({'a':q,'b':q},seed=3);s.update_losses({'a':1,'b':4},decay=0.0);state=s.state_dict();t=AdaptiveKnowledgeSampler({'a':q,'b':q},seed=99);t.load_state_dict(state);assert t.weights==s.weights and t.loss_ema==s.loss_ema


def test_sampler_resume_reproduces_next_batch(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 q=tmp_path/'a.jsonl';q.write_text('['+','.join(map(str,range(20)))+']\n');a=AdaptiveKnowledgeSampler({'a':q,'b':q},seed=4);a.sample(3,10);st=a.state_dict();expected=a.sample(3,10);b=AdaptiveKnowledgeSampler({'a':q,'b':q},seed=99);b.load_state_dict(st);assert b.sample(3,10)==expected


def test_sampler_deduplicates_repeated_rows(tmp_path):
 from reference.hz0i_knowledge_sampler import KnowledgeDenseSampler
 q=tmp_path/'d.jsonl';q.write_text('[1,2]\n[1,2]\n[2,3]\n');s=KnowledgeDenseSampler({'d':q});assert len(s.domains['d'])==2


def test_adaptive_sampler_preserves_domain_floor(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 q=tmp_path/'d.jsonl';q.write_text('[1,2,3]\n');s=AdaptiveKnowledgeSampler({'a':q,'b':q,'c':q},min_weight=.1);s.update_losses({'a':1,'b':1,'c':100},decay=0.0);assert min(s.weights.values())>=.1


def test_adaptive_sampler_checkpoint_restores_temperature(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 q=tmp_path/'a.jsonl';q.write_text('[1,2,3]\n')
 paths={'x':q,'y':q};a=AdaptiveKnowledgeSampler(paths,seed=3,temperature=1.7,min_weight=.1)
 state=a.state_dict();b=AdaptiveKnowledgeSampler(paths,seed=9,temperature=.2,min_weight=.2);b.load_state_dict(state);assert b.temperature==1.7 and b.min_weight==.1


def test_sampler_checkpoint_rejects_domain_mismatch(tmp_path):
 from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
 q=tmp_path/'a.jsonl';q.write_text('[1,2]\n')
 state=AdaptiveKnowledgeSampler({'a':q,'b':q}).state_dict()
 try: AdaptiveKnowledgeSampler({'a':q,'c':q}).load_state_dict(state)
 except ValueError: pass
 else: raise AssertionError('domain mismatch must be rejected')
