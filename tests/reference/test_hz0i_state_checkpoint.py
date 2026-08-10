import torch
from reference.hz0i_state_storage import quantize_int8_per_head
from reference.hz0i_state_checkpoint import save_state_checkpoint,load_state_checkpoint
def test_state_checkpoint_roundtrip(tmp_path):
 s=[torch.randn(1,4,8,16),torch.randn(1,4,8,16)];p=tmp_path/'s.pt';save_state_checkpoint(p,s,128,model_fingerprint='x');r,pos,f=load_state_checkpoint(p);assert pos==128 and f=='x';assert all(torch.equal(a,b) for a,b in zip(s,r))


def test_checkpoint_resume_factorized_stream(tmp_path):
 from reference.hz0i_factorized_bdh import FactorizedBDH,factorized_stream_chunk
 from reference.hz0i_bdh_model import HZ0IBDHConfig
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3).eval();x=torch.randint(0,32,(1,12));H=c.n_head;N=c.n_embd*c.mlp_internal_dim_multiplier//H;st=[torch.zeros(1,H,N,c.n_embd) for _ in range(c.n_layer)];st,a=factorized_stream_chunk(m,st,x[:,:5],0);p=tmp_path/'resume.pt';save_state_checkpoint(p,st,5);st,pos,_=load_state_checkpoint(p);st,b=factorized_stream_chunk(m,st,x[:,5:],pos);assert torch.isfinite(torch.cat([a,b],1)).all()


def test_quantized_state_checkpoint_roundtrip(tmp_path):
 from reference.hz0i_state_storage import quantize_int8_per_head
 x=quantize_int8_per_head(torch.randn(1,4,8,6));p=tmp_path/'q.pt';save_state_checkpoint(p,[x],4);r,pos,_=load_state_checkpoint(p);assert r[0].values.dtype==torch.int8 and pos==4


def test_quantized_checkpoint_resumes_factorized_stream(tmp_path):
 from reference.hz0i_factorized_bdh import FactorizedBDH,factorized_stream_chunk,init_factorized_states,factorized_stream_sequence
 from reference.hz0i_bdh_model import HZ0IBDHConfig
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3).eval();x=torch.randint(0,32,(1,12));st=init_factorized_states(m,1);st,a=factorized_stream_chunk(m,st,x[:,:4],0);st=[quantize_int8_per_head(z) for z in st];p=tmp_path/'qresume.pt';save_state_checkpoint(p,st,4);st,pos,_=load_state_checkpoint(p);st,b=factorized_stream_sequence(m,x[:,4:],[8],states=st,state_storage='int8_head');assert torch.isfinite(torch.cat([a,b],1)).all()
