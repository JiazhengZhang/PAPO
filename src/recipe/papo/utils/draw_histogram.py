#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot value-frequency histograms from entropy_stats.jsonl.

Supported JSONL record structure (examples):
1) Raw samples:
   {
     "step": 100,
     "t12": {
       "hist": [0.1, -0.2, 0.3, ...]
     }
   }

2) Pre-binned histogram:
   {
     "step": 100,
     "t12": {
       "hist": {
         "counts": [...],
         "bin_edges": [...]
       }
     }
   }

3) Alternative pre-binned form:
   {
     "step": 100,
     "t12": {
       "hist": {
         "hist": [...],
         "edges": [...]
       }
     }
   }

Usage:
python plot_entropy_hist.py \
    --input-dir /path/to/dump_dir \
    --metric t12 \
    --step-min 100 \
    --step-max 300 \
    --mode aggregate \
    --out t12_hist.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing entropy_stats.jsonl")
    parser.add_argument("--metric", type=str, default="t12", choices=["t1", "t2", "t12"])
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="aggregate",
        choices=["aggregate", "overlay"],
        help="aggregate: merge selected steps into one histogram; overlay: draw one histogram per step",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of bins for raw-sample histogram plotting")
    parser.add_argument("--max-steps-overlay", type=int, default=20, help="Max number of steps to plot in overlay mode")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output image path; default is auto-generated")
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=True,
        help="Use log scale on frequency axis (default: True)",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        default=False,
        help="Force linear scale on frequency axis",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip malformed JSON at line {line_no}: {e}")
    return records


def in_step_range(step: int, step_min: Optional[int], step_max: Optional[int]) -> bool:
    if step_min is not None and step < step_min:
        return False
    if step_max is not None and step > step_max:
        return False
    return True


def extract_hist_payload(record: Dict[str, Any], metric: str) -> Any:
    """
    Accept either:
      record["t12"]["hist"]
    or flattened:
      record["t12"] already the hist payload
    """
    metric_obj = record.get(metric)
    if metric_obj is None:
        return None

    if isinstance(metric_obj, dict) and "hist" in metric_obj:
        return metric_obj["hist"]
    return metric_obj


def normalize_hist_payload(payload: Any) -> Tuple[str, Any]:
    """
    Return:
      ("raw", np.ndarray)
      ("binned", (counts, edges))
    """
    if payload is None:
        raise ValueError("Empty hist payload")

    # Case 1: raw list / array
    if isinstance(payload, list):
        arr = np.asarray(payload, dtype=float).reshape(-1)
        return "raw", arr

    # Case 2: dict with counts / bin_edges
    if isinstance(payload, dict):
        if "counts" in payload and "bin_edges" in payload:
            counts = np.asarray(payload["counts"], dtype=float)
            edges = np.asarray(payload["bin_edges"], dtype=float)
            return "binned", (counts, edges)

        if "hist" in payload and "edges" in payload:
            counts = np.asarray(payload["hist"], dtype=float)
            edges = np.asarray(payload["edges"], dtype=float)
            return "binned", (counts, edges)

        if "values" in payload:
            arr = np.asarray(payload["values"], dtype=float).reshape(-1)
            return "raw", arr

    raise ValueError(f"Unsupported hist payload format: {type(payload)} -> {payload if isinstance(payload, dict) else '...'}")


def plot_aggregate_raw(all_arrays: List[np.ndarray], bins: int, title: str, out: Path, use_log_y: bool) -> None:
    merged = np.concatenate(all_arrays, axis=0)
    merged = merged[np.isfinite(merged)]

    plt.figure(figsize=(9, 5))
    plt.hist(merged, bins=bins)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    _apply_axis_style(use_log_y)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_overlay_raw(step_to_arr: List[Tuple[int, np.ndarray]], bins: int, title: str, out: Path, use_log_y: bool) -> None:
    plt.figure(figsize=(10, 6))
    for step, arr in step_to_arr:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        plt.hist(arr, bins=bins, histtype="step", linewidth=1.5, label=f"step={step}")

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    _apply_axis_style(use_log_y)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_aggregate_binned(all_binned: List[Tuple[np.ndarray, np.ndarray]], title: str, out: Path, use_log_y: bool) -> None:
    ref_edges = all_binned[0][1]
    total_counts = np.zeros_like(all_binned[0][0], dtype=float)

    for counts, edges in all_binned:
        if len(edges) != len(ref_edges) or not np.allclose(edges, ref_edges):
            raise ValueError("Binned histograms do not share identical bin edges; cannot aggregate directly.")
        total_counts += counts

    centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
    widths = np.diff(ref_edges)

    plt.figure(figsize=(9, 5))
    plt.bar(centers, total_counts, width=widths, align="center")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    _apply_axis_style(use_log_y)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_overlay_binned(step_to_binned: List[Tuple[int, Tuple[np.ndarray, np.ndarray]]], title: str, out: Path, use_log_y: bool) -> None:
    plt.figure(figsize=(10, 6))

    for step, (counts, edges) in step_to_binned:
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.step(centers, counts, where="mid", label=f"step={step}")

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    _apply_axis_style(use_log_y)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def _apply_axis_style(use_log_y: bool) -> None:
    if use_log_y:
        plt.yscale("log")


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.input_dir) / "entropy_stats.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Cannot find file: {jsonl_path}")
    use_log_y = args.log_y and not args.linear_y
    records = load_jsonl(jsonl_path)
    selected: List[Tuple[int, str, Any]] = []

    for rec in records:
        step = rec.get("step")
        if step is None:
            continue
        if not in_step_range(step, args.step_min, args.step_max):
            continue

        payload = extract_hist_payload(rec, args.metric)
        if payload is None:
            continue

        try:
            mode, normed = normalize_hist_payload(payload)
            selected.append((step, mode, normed))
        except Exception as e:
            print(f"[WARN] Skip step={step}: {e}")

    if not selected:
        raise RuntimeError("No valid histogram data found in the requested step range.")

    # Keep stable order
    selected.sort(key=lambda x: x[0])

    modes = {m for _, m, _ in selected}
    if len(modes) != 1:
        raise RuntimeError(f"Mixed histogram payload formats detected in selected steps: {modes}")
    hist_mode = next(iter(modes))

    title = args.title
    if title is None:
        if args.step_min is None and args.step_max is None:
            step_desc = "all steps"
        else:
            step_desc = f"steps [{args.step_min}, {args.step_max}]"
        title = f"{args.metric} histogram ({step_desc}, mode={args.mode})"

    if args.out is None:
        out_name = f"{args.metric}_{args.mode}"
        if args.step_min is not None or args.step_max is not None:
            out_name += f"_{args.step_min}_{args.step_max}"
        args.out = str(Path(args.input_dir) / f"{out_name}.png")

    out_path = Path(args.out)

    if args.mode == "overlay" and len(selected) > args.max_steps_overlay:
        print(
            f"[INFO] overlay mode selected, but {len(selected)} steps found. "
            f"Only plotting the last {args.max_steps_overlay} steps."
        )
        selected = selected[-args.max_steps_overlay :]

    if hist_mode == "raw":
        step_to_arr = [(step, data) for step, _, data in selected]
        if args.mode == "aggregate":
            plot_aggregate_raw([arr for _, arr in step_to_arr], args.bins, title, out_path, use_log_y)
        else:
            plot_overlay_raw(step_to_arr, args.bins, title, out_path, use_log_y)

    elif hist_mode == "binned":
        step_to_binned = [(step, data) for step, _, data in selected]
        if args.mode == "aggregate":
            plot_aggregate_binned([b for _, b in step_to_binned], title, out_path, use_log_y)
        else:
            plot_overlay_binned(step_to_binned, title, out_path, use_log_y)

    print(f"[OK] Saved figure to: {out_path}")


if __name__ == "__main__":
    main()