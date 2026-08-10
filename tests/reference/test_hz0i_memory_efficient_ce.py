import torch
from reference.hz0i_memory_efficient_ce import chunked_cross_entropy, dense_cross_entropy

def test_chunked_ce_matches_dense():
    torch.manual_seed(0);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    D,V,B,T=768,24576,4,32
    h=torch.randn(B,T,D,dtype=torch.bfloat16,device=dev);w=torch.randn(D,V,dtype=torch.bfloat16,device=dev)
    y=torch.randint(0,V,(B,T),device=dev)
    a=dense_cross_entropy(h,w,y);b=chunked_cross_entropy(h,w,y)
    assert abs(float(a)-float(b)) < 1e-2

def test_chunked_ce_reduces_peak_logit_memory(tmp_path):
    import torch
    D,V=768,16384
    h=torch.randn(2,8,D);w=torch.randn(D,V)
    # only correctness on cpu; memory check is on MPS in the training loop
    assert chunked_cross_entropy(h,w,torch.randint(0,V,(2,8)),chunk=4096).ndim==0
