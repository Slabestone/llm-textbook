#!/usr/bin/env python3
"""Regenerate and verify textbook Figure 1.17 against the public training code.

Use only Python's standard library. No Mac-only paths, network, or pretrained weights.
The canonical numerical experiment lives in llm_vol1_ch10_bigram_train_eval_v01.py;
this script does NOT invent or fit a replacement curve.
"""
import argparse
import contextlib
import io
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "verify/chapter1/llm_vol1_ch10_bigram_train_eval_v01.py"
DATA = ROOT / "data/llm_vol1_ch10_learning_curves_v01.dat"


def run_original_experiment():
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        runpy.run_path(str(SOURCE), run_name="__main__")
    result = json.loads(capture.getvalue())
    if result.get("status") != "PASS":
        raise AssertionError("Original experiment did not return PASS")
    return result


def format_14_significant_digits(value):
    """Match the original pgfplots .dat file's fixed 14-significant-digit format."""
    if value == 0:
        return "0.0000000000000"
    decimals = max(0, 13 - math.floor(math.log10(abs(value))))
    return f"{value:.{decimals}f}"


def render_data(rows):
    lines = ["step train validation"]
    for row in rows:
        lines.append(" ".join(format_14_significant_digits(row[key])
                              for key in ("step", "train", "validation")))
    return "\n".join(lines) + "\n"


def verify(result, rendered):
    rows = result["series"]
    stopping = result["early_stopping"]
    if [row["step"] for row in rows] != list(range(121)):
        raise AssertionError("Expected exactly 121 consecutive steps (0..120)")
    if (stopping["selected_step"], stopping["stop_step"]) != (23, 35):
        raise AssertionError(f"Early-stop regression: {stopping!r}")
    actual_min_step = min(rows, key=lambda row: row["validation"])["step"]
    if actual_min_step != 24:
        raise AssertionError(f"Absolute validation minimum changed: {actual_min_step}")
    # The validation improvement at 24 is smaller than min_delta=1e-4;
    # therefore the rule retains step 23 despite the slightly lower raw loss.
    gain = rows[23]["validation"] - rows[24]["validation"]
    if not (0 < gain < stopping["min_delta"]):
        raise AssertionError(f"min_delta teaching example invalid: {gain}")
    if not (rows[120]["train"] < rows[35]["train"]
            and rows[120]["validation"] > rows[35]["validation"]):
        raise AssertionError("Counterfactual continuation no longer illustrates overfitting")
    if result["fd_gradient_max_error"] >= 1e-9:
        raise AssertionError("Finite-difference gradient audit failed")
    actual = DATA.read_text(encoding="utf-8")
    if rendered != actual:
        for line_no, (expected, found) in enumerate(
                zip(rendered.splitlines(), actual.splitlines()), start=1):
            if expected != found:
                raise AssertionError(
                    f"Committed Figure 1.17 table differs at line {line_no}: "
                    f"generated={expected!r}, committed={found!r}")
        raise AssertionError("Committed Figure 1.17 table differs in length/termination")
    return gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-dat", type=Path, metavar="OUTPUT",
        help="Optionally save regenerated .dat to OUTPUT; checked-in file remains untouched",
    )
    args = parser.parse_args()
    result = run_original_experiment()
    rendered = render_data(result["series"])
    gain = verify(result, rendered)
    if args.write_dat:
        args.write_dat.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "canonical_code": SOURCE.relative_to(ROOT).as_posix(),
        "checked_data": DATA.relative_to(ROOT).as_posix(),
        "data_rows": len(result["series"]),
        "rule_selected_checkpoint": result["early_stopping"]["selected_step"],
        "early_stop_trigger": result["early_stopping"]["stop_step"],
        "absolute_validation_minimum": 24,
        "step24_improvement_below_min_delta": gain,
        "continued_reference_end": result["series"][-1]["step"],
        "train_loss_at_120": result["series"][-1]["train"],
        "validation_loss_at_120": result["series"][-1]["validation"],
        "numeric_table": "byte-exact match",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
