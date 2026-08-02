from scripts.hz0a_gdn2_fix_recurrence_benchmark import benchmark


def test_gdn2_fix_benchmark_is_finite_and_reports_all_controls():
    report = benchmark(seed=3, trials=4, dim=8)
    assert report["finite"]
    assert set(report["results"]) == {"old", "kda", "gdn2_fix", "attention"}
    assert all(item["total_cases"] == 12 for item in report["results"].values())
