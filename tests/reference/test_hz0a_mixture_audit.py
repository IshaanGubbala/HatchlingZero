from pathlib import Path

from scripts.hz0a_audit_mixture import audit


def test_a5_mixture_manifest_audits_real_sources() -> None:
    report = audit(Path("data/hz0a_mixture_manifest.json"))
    assert report["all_sources_hash_consistent"] is True
    assert report["grand_total_tokens"] > 0
    names = {source["name"] for source in report["sources"]}
    assert names == {
        "wikitext-103",
        "hz0a_local_repo_corpus",
        "codeparrot_codeparrot-clean-valid",
        "codeparrot_github-jupyter-text-code-pairs",
        "open-web-math",
    }
    # Honest gates: code and math must be at real, near-target absolute scale
    # (not padded/repeated), and json/terminal must stay visibly small since
    # no non-gated large-scale source was found for either -- this must
    # never silently claim those two categories reached target scale.
    pct = report["actual_mixture_pct_of_grand_total"]
    assert report["category_token_totals"]["code"] >= 35_000_000
    assert report["category_token_totals"]["mathematical_and_structured"] >= 5_000_000
    assert report["category_token_totals"]["documentation"] >= 9_000_000
    assert pct["json_and_configuration"] < 1
    assert pct["terminal_and_debugging"] < 1
