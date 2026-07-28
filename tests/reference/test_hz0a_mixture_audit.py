from pathlib import Path

from scripts.hz0a_audit_mixture import audit


def test_a5_mixture_manifest_audits_real_sources() -> None:
    report = audit(Path("data/hz0a_mixture_manifest.json"))
    assert report["all_sources_hash_consistent"] is True
    assert report["grand_total_tokens"] > 0
    names = {source["name"] for source in report["sources"]}
    assert names == {"wikitext-103", "hz0a_local_repo_corpus"}
    # Honest gate: this must never silently claim the plan's 40/35/10/5/5/5
    # target is met when the underlying corpus is still local-repo scale.
    assert report["actual_mixture_pct_of_grand_total"]["general_text"] > 90
