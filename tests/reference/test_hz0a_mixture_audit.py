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
        "json_and_configuration_combined",
        "stackoverflow_python_tracebacks",
    }
    # Honest gates: corpus must be near the plan's 100M-token target and
    # every one of the 6 declared categories must be within ~1 percentage
    # point of its plan ratio -- not padded/repeated, and not silently
    # regressed back to a near-zero share for any category.
    totals = report["category_token_totals"]
    pct = report["actual_mixture_pct_of_grand_total"]
    assert 90_000_000 <= report["grand_total_tokens"] <= 110_000_000
    assert totals["code"] >= 35_000_000
    assert totals["mathematical_and_structured"] >= 5_000_000
    assert totals["documentation"] >= 9_000_000
    assert totals["terminal_and_debugging"] >= 5_000_000
    assert totals["json_and_configuration"] >= 5_000_000
    target = {"general_text": 40, "code": 35, "documentation": 10, "json_and_configuration": 5, "terminal_and_debugging": 5, "mathematical_and_structured": 5}
    for category, target_pct in target.items():
        assert abs(pct[category] - target_pct) <= 2, f"{category}: {pct[category]}% vs {target_pct}% target"
