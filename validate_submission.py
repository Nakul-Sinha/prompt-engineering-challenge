"""Strict validator for ./working/submission.csv against the challenge schema."""
import csv
import json
from pathlib import Path

csv.field_size_limit(10**8)
ROOT = Path("./dataset/public")
SUB = Path("./working/submission.csv")
FIELDS = ["source_token", "name_type_token", "library_token"]
OPTIONS_KEY = {"source_token": "source_options",
               "name_type_token": "name_type_options",
               "library_token": "library_options"}


def load_test_options(path):
    opts = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            opts[row["id"]] = {f: set(json.loads(row[OPTIONS_KEY[f]])) for f in FIELDS}
    return opts


def main():
    sample = list(csv.DictReader(open(ROOT / "sample_submission.csv", encoding="utf-8")))
    sub = list(csv.DictReader(open(SUB, encoding="utf-8")))
    opts = load_test_options(ROOT / "test.csv")

    assert list(sub[0].keys())[:2] == ["id", "answer_json"], sub[0].keys()
    sample_ids = {r["id"] for r in sample}
    sub_ids = [r["id"] for r in sub]
    assert len(sub_ids) == len(set(sub_ids)), "duplicate ids"
    assert set(sub_ids) == sample_ids, "id set mismatch with sample_submission"

    bad = 0
    for r in sub:
        obj = json.loads(r["answer_json"])  # must be valid JSON
        assert set(obj.keys()) == set(FIELDS), (r["id"], obj.keys())
        for f in FIELDS:
            assert isinstance(obj[f], str), (r["id"], f)
            if r["id"] in opts and obj[f] not in opts[r["id"]][f]:
                bad += 1
    assert bad == 0, f"{bad} chosen tokens not in that row's candidate options"
    print(f"Submission validation PASSED: {len(sub)} rows, all tokens in-options, valid JSON.")


if __name__ == "__main__":
    main()
