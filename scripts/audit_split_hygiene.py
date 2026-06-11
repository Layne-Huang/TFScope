#!/usr/bin/env python
"""Audit split and retrieval artifacts for identity leakage."""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tfscope.data.split_hygiene import (
    SPLIT_NAMES,
    audit_split,
    donor_exclusion_reasons,
    specific_source_ids,
    summarize_split,
    write_json,
)


def audit_index(df: pd.DataFrame, split: dict, index: dict) -> dict:
    rows = df.set_index("filename", drop=False)
    violations = []
    reason_counts = Counter()
    train_set = set(split["train"])
    identity_source_ids = specific_source_ids(df)

    for split_name in SPLIT_NAMES:
        for query_filename in split.get(split_name, []):
            if query_filename not in rows.index:
                continue
            for rank, item in enumerate(index.get(query_filename, []), start=1):
                donor_filename = item["nn_filename"]
                if donor_filename not in rows.index:
                    violations.append(
                        {
                            "query": query_filename,
                            "donor": donor_filename,
                            "rank": rank,
                            "reason": "unknown_donor",
                        }
                    )
                    reason_counts["unknown_donor"] += 1
                    continue
                if donor_filename not in train_set:
                    violations.append(
                        {
                            "query": query_filename,
                            "donor": donor_filename,
                            "rank": rank,
                            "reason": "donor_not_train",
                        }
                    )
                    reason_counts["donor_not_train"] += 1
                for reason in donor_exclusion_reasons(
                    rows.loc[query_filename],
                    rows.loc[donor_filename],
                    identity_source_ids=identity_source_ids,
                ):
                    violations.append(
                        {
                            "query": query_filename,
                            "donor": donor_filename,
                            "rank": rank,
                            "reason": reason,
                        }
                    )
                    reason_counts[reason] += 1

    return {
        "clean": not violations,
        "n_violations": len(violations),
        "reason_counts": dict(reason_counts),
        "violations": violations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--index")
    parser.add_argument("--out")
    parser.add_argument("--fail-on-leakage", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    with open(args.split) as handle:
        split = json.load(handle)

    report = {
        "split": audit_split(df, split).to_dict(),
        "summary": summarize_split(df, split),
    }
    if args.index:
        with open(args.index) as handle:
            index = json.load(handle)
        report["retrieval_index"] = audit_index(df, split, index)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        write_json(args.out, report)

    clean = report["split"]["clean"] and report.get(
        "retrieval_index", {"clean": True}
    )["clean"]
    if args.fail_on_leakage and not clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
