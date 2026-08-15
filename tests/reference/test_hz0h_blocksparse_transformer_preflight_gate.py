import json
from scripts.hz0h_blocksparse_transformer_preflight_gate import evaluate

def report(speed=1.31,ram=.69):
 return {'device':'cuda','parameter_ratio_to_transformer':1.003,'blocksparse':{'finite_loss':True,'finite_gradients':True},'matched_rope_transformer':{'finite_loss':True,'finite_gradients':True},'blocksparse_over_transformer_speed_ratio':speed,'blocksparse_over_transformer_peak_memory_ratio':ram}
def test_passes_all_strict_systems_checks():
 assert evaluate(report())['systems_preflight_pass']
def test_rejects_each_target_threshold():
 assert not evaluate(report(speed=1.29))['systems_preflight_pass']
 assert not evaluate(report(ram=.71))['systems_preflight_pass']
