#!/usr/bin/env python3
"""Strict systems gate for the matched BlockBDH/Transformer synthetic preflight."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def evaluate(report: dict) -> dict:
    ratio=report.get('parameter_ratio_to_transformer')
    checks={'cuda_report':report.get('device')=='cuda','parameter_match':isinstance(ratio,(int,float)) and .9901<=ratio<=1.01,'finite_blocksparse_step':bool(report.get('blocksparse',{}).get('finite_loss')) and bool(report.get('blocksparse',{}).get('finite_gradients')),'finite_transformer_step':bool(report.get('matched_rope_transformer',{}).get('finite_loss')) and bool(report.get('matched_rope_transformer',{}).get('finite_gradients'))}
    speed=report.get('blocksparse_over_transformer_speed_ratio');ram=report.get('blocksparse_over_transformer_peak_memory_ratio')
    checks['speed']=isinstance(speed,(int,float)) and speed>=1.30
    checks['ram']=isinstance(ram,(int,float)) and ram<=.70
    return {'checks':checks,'speed_ratio':speed,'peak_memory_ratio':ram,'parameter_ratio':ratio,'systems_preflight_pass':all(checks.values()),'claim_eligible':False,'reason':'untrained synthetic optimizer-step screen only; trained quality, multi-seed stability, and frozen evaluation remain mandatory'}
def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args();result=evaluate(json.loads(a.report.read_text()));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['systems_preflight_pass'] else 2)
if __name__=='__main__':main()
