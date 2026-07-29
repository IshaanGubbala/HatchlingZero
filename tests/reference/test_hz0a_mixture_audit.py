from pathlib import Path

from scripts.hz0a_audit_mixture import audit


def test_a5_mixture_manifest_audits_real_sources() -> None:
    report = audit(Path("data/hz0a_mixture_manifest.json"))
    assert report["all_sources_hash_consistent"] is True
    assert report["grand_total_tokens"] > 0
    names = {source["name"] for source in report["sources"]}
    assert names == {
        "wikitext-103-subsampled",
        "hz0a_local_repo_corpus",
        "codeparrot_codeparrot-clean-valid",
        "codeparrot_github-jupyter-text-code-pairs",
        "open-web-math",
        "glaive-function-calling-v2",
        "stackoverflow_python_tracebacks",
    }
    # Honest gates: corpus must be near the plan's 100M-token target and
    # every category must be within real, hash-verified striking distance
    # of its declared ratio -- not padded/repeated, and not silently
    # regressed back to a near-zero non-text share.
    totals = report["category_token_totals"]
    pct = report["actual_mixture_pct_of_grand_total"]
    assert 90_000_000 <= report["grand_total_tokens"] <= 110_000_000
    assert totals["code"] >= 35_000_000
    assert totals["mathematical_and_structured"] >= 5_000_000
    assert totals["documentation"] >= 9_000_000
    assert totals["terminal_and_debugging"] >= 5_000_000
    assert totals["json_and_configuration"] >= 3_000_000
    assert 35 <= pct["general_text"] <= 45
    assert 30 <= pct["code"] <= 40
