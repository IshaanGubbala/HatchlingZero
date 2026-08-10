import torch
from reference.hz0i_state_storage import quantize_int8,dequantize_int8,relative_error
def test_int8_state_roundtrip_is_finite_and_bounded():
 torch.manual_seed(2);x=torch.randn(2,4,32,16);q=quantize_int8(x);y=dequantize_int8(q);assert q.values.dtype==torch.int8;assert torch.isfinite(y).all();assert relative_error(x,q)<.02
def test_state_storage_preserves_shape():
 x=torch.randn(3,5);q=quantize_int8(x);assert dequantize_int8(q).shape==x.shape


def test_per_head_int8_roundtrip():
 from reference.hz0i_state_storage import quantize_int8_per_head
 x=torch.randn(2,4,16,8);q=quantize_int8_per_head(x);assert q.scale.shape==(2,4,1,1);assert relative_error(x,q)<.02
