"""
Analyze SQA evaluation results by SQA3D paper question type categories.

Categories (per SQA3D paper):
  What, Is, How many, Can, Which, Others

Uses the 'contains' metric (correct_contains) to measure performance.

Usage:
    python analyze_by_question_type.py <results_json> [--save]
    python analyze_by_question_type.py sqa_test_results_cot_video_qwen3_vl_2b_instruct_all_scenes_0307.json --save
    
    # Analyze multiple result files side by side
    python analyze_by_question_type.py file1.json file2.json --save
"""

import json
import sys
import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path


def classify_question_type(question: str) -> str:
    """
    Classify question into SQA3D paper categories by first word(s).

    Categories: What, Is, How many, Can, Which, Others
    """
    text = question.strip().lower()
    if text.startswith("what"):
        return "What"
    elif text.startswith("is "):
        return "Is"
    elif text.startswith("how many") or text.startswith("how much"):
        return "How many"
    elif text.startswith("can"):
        return "Can"
    elif text.startswith("which"):
        return "Which"
    else:
        return "Others"


CATEGORY_ORDER = ["What", "Is", "How many", "Can", "Which", "Others"]


def analyze_results(results_file: str) -> dict:
    """Analyze a single results JSON file and return per-category stats."""
    with open(results_file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"  ⚠️  No results found in {results_file}")
        return {}

    # Accumulate per-category
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "exact_correct": 0})

    for r in results:
        question = r.get("question", "")
        cat = classify_question_type(question)

        stats[cat]["total"] += 1

        # Contains metric
        if r.get("correct_contains", False):
            stats[cat]["correct"] += 1

        # Also track exact match for reference
        if r.get("correct", False):
            stats[cat]["exact_correct"] += 1

    return dict(stats)


def print_table(stats: dict, title: str = ""):
    """Print a formatted results table."""
    if title:
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")

    # Header
    print(
        f"{'Category':<12} | {'Contains Acc':>12} | {'Correct':>8} | {'Total':>6} | {'Exact Match':>12}"
    )
    print("-" * 72)

    total_contains_correct = 0
    total_exact_correct = 0
    total_count = 0

    for cat in CATEGORY_ORDER:
        s = stats.get(cat, {"correct": 0, "total": 0, "exact_correct": 0})
        total = s["total"]
        contains_correct = s["correct"]
        exact_correct = s["exact_correct"]

        contains_acc = contains_correct / total * 100 if total > 0 else 0.0
        exact_acc = exact_correct / total * 100 if total > 0 else 0.0

        print(
            f"{cat:<12} | {contains_acc:>11.2f}% | {contains_correct:>8d} | {total:>6d} | {exact_acc:>11.2f}%"
        )

        total_contains_correct += contains_correct
        total_exact_correct += exact_correct
        total_count += total

    # Overall
    print("-" * 72)
    overall_contains = (
        total_contains_correct / total_count * 100 if total_count > 0 else 0.0
    )
    overall_exact = (
        total_exact_correct / total_count * 100 if total_count > 0 else 0.0
    )
    print(
        f"{'Overall':<12} | {overall_contains:>11.2f}% | {total_contains_correct:>8d} | {total_count:>6d} | {overall_exact:>11.2f}%"
    )
    print()


def save_csv(all_stats: dict, output_path: str):
    """Save results to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        if len(all_stats) == 1:
            # Single file mode
            name = list(all_stats.keys())[0]
            stats = all_stats[name]
            writer.writerow(
                [
                    "Category",
                    "Contains Accuracy (%)",
                    "Contains Correct",
                    "Total",
                    "Exact Match Accuracy (%)",
                    "Exact Match Correct",
                ]
            )

            total_cc = 0
            total_ec = 0
            total_n = 0
            for cat in CATEGORY_ORDER:
                s = stats.get(cat, {"correct": 0, "total": 0, "exact_correct": 0})
                total = s["total"]
                cc = s["correct"]
                ec = s["exact_correct"]
                c_acc = cc / total * 100 if total > 0 else 0.0
                e_acc = ec / total * 100 if total > 0 else 0.0
                writer.writerow([cat, f"{c_acc:.2f}", cc, total, f"{e_acc:.2f}", ec])
                total_cc += cc
                total_ec += ec
                total_n += total

            ov_c = total_cc / total_n * 100 if total_n > 0 else 0.0
            ov_e = total_ec / total_n * 100 if total_n > 0 else 0.0
            writer.writerow(
                ["Overall", f"{ov_c:.2f}", total_cc, total_n, f"{ov_e:.2f}", total_ec]
            )
        else:
            # Multi-file comparison mode
            names = list(all_stats.keys())
            header = ["Category"]
            for name in names:
                short = Path(name).stem
                header.extend([f"{short} Contains(%)", f"{short} Correct", f"{short} Total"])
            writer.writerow(header)

            for cat in CATEGORY_ORDER + ["Overall"]:
                row = [cat]
                for name in names:
                    stats = all_stats[name]
                    if cat == "Overall":
                        cc = sum(
                            stats.get(c, {}).get("correct", 0) for c in CATEGORY_ORDER
                        )
                        n = sum(
                            stats.get(c, {}).get("total", 0) for c in CATEGORY_ORDER
                        )
                    else:
                        s = stats.get(
                            cat, {"correct": 0, "total": 0, "exact_correct": 0}
                        )
                        cc = s["correct"]
                        n = s["total"]
                    acc = cc / n * 100 if n > 0 else 0.0
                    row.extend([f"{acc:.2f}", cc, n])
                writer.writerow(row)

    print(f"💾 Results saved to: {output_path}")


def save_json(all_stats: dict, output_path: str):
    """Save results to JSON file."""
    output = {}
    for name, stats in all_stats.items():
        file_output = {"per_category": {}, "overall": {}}
        total_cc = 0
        total_ec = 0
        total_n = 0
        for cat in CATEGORY_ORDER:
            s = stats.get(cat, {"correct": 0, "total": 0, "exact_correct": 0})
            total = s["total"]
            cc = s["correct"]
            ec = s["exact_correct"]
            c_acc = cc / total if total > 0 else 0.0
            e_acc = ec / total if total > 0 else 0.0
            file_output["per_category"][cat] = {
                "contains_accuracy": round(c_acc, 4),
                "contains_correct": cc,
                "exact_match_accuracy": round(e_acc, 4),
                "exact_match_correct": ec,
                "total": total,
            }
            total_cc += cc
            total_ec += ec
            total_n += total

        file_output["overall"] = {
            "contains_accuracy": round(total_cc / total_n, 4) if total_n > 0 else 0.0,
            "contains_correct": total_cc,
            "exact_match_accuracy": round(total_ec / total_n, 4) if total_n > 0 else 0.0,
            "exact_match_correct": total_ec,
            "total": total_n,
        }
        output[Path(name).stem] = file_output

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SQA results by SQA3D paper question type categories (What, Is, How many, Can, Which, Others)"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="One or more result JSON files to analyze",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to CSV and JSON files",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (default: auto-generated from input filename)",
    )
    args = parser.parse_args()

    all_stats = {}

    for results_file in args.results_files:
        if not os.path.exists(results_file):
            print(f"❌ File not found: {results_file}")
            continue

        print(f"\n📊 Analyzing: {results_file}")
        stats = analyze_results(results_file)
        if stats:
            short_name = Path(results_file).stem
            print_table(stats, title=short_name)
            all_stats[results_file] = stats

    if not all_stats:
        print("No results to save.")
        return

    # Save
    if args.save:
        if args.output_prefix:
            prefix = args.output_prefix
        elif len(args.results_files) == 1:
            prefix = Path(args.results_files[0]).stem + "_by_question_type"
        else:
            prefix = "comparison_by_question_type"

        save_csv(all_stats, f"{prefix}.csv")
        save_json(all_stats, f"{prefix}.json")


if __name__ == "__main__":
    main()
