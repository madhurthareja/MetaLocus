"""Runs the real A/B/C/D eval harness (src/locus/eval/) across every synthetic
scenario and dumps the results for the dashboard's Experiment Lab page.

Only reports metrics the harness actually computes: answer rate (split
in-view / out-of-view), mean bearing error, mean range error. UI.md's
Experiment Lab wishlist also names hazard-detection rate, false
warnings/missed hazards, track fragmentation, re-association performance and
TTC prediction performance -- none of those are implemented anywhere in
src/locus/eval/ today, so rather than inventing a methodology for them here,
this script records them as not_implemented with a pointer to what would be
needed. Don't manufacture a number for a metric that doesn't exist.

Usage: python scripts/generate_ablation_results.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from locus.data import synthetic
from locus.eval.conditions import ConditionA, ConditionB, ConditionC, ConditionD
from locus.eval.harness import run_condition
from locus.eval.questions import generate_questions

SCENARIOS = {
    "straight_walk": synthetic.straight_walk,
    "head_turn_away": synthetic.head_turn_away,
    "head_turn_return": synthetic.head_turn_return,
    "cluttered_occlusion": synthetic.cluttered_occlusion,
    "crossing_hazard": synthetic.crossing_hazard,
}

CONDITIONS = {
    "A_stateless": {
        "label": "A: Stateless baseline",
        "description": "Zero memory. Forgets every object the instant it is not in the current frame's detections.",
        "make": lambda intr: ConditionA(intr),
    },
    "B_window": {
        "label": "B: Sliding-window baseline",
        "description": "Keeps the last 1.0s of detections per concept (survives a brief glance away) but forgets for good once the window elapses. No notion that an object is 'still out there'.",
        "make": lambda intr: ConditionB(intr, window_s=1.0),
    },
    "C_box": {
        "label": "C: Persistent memory, box-lifted",
        "description": "Full persistent SpatialMemoryGraph (decay, eviction, spatial re-association) but instances are lifted to 3D from the detection's bounding box, not its segmentation mask.",
        "make": lambda intr: ConditionC(intr),
    },
    "D_mask": {
        "label": "D: Full system (persistent memory, mask-lifted)",
        "description": "Identical persistent-memory pipeline as C; instances are lifted from the real SAM segmentation mask (median masked depth), not the box. Isolates mask-vs-box lifting noise as the only variable vs. C.",
        "make": lambda intr: ConditionD(intr),
    },
}

NOT_IMPLEMENTED = {
    "hazard_detection_rate": "No ground-truth hazard-event labels or hazard-trigger scoring exist in src/locus/eval/ -- only per-question bearing/range answerability is scored.",
    "false_warnings": "Same reason: hazard warnings aren't a scored event type in the harness.",
    "missed_hazards": "Same reason.",
    "track_fragmentation": "No track-continuity/ID-switch metric is computed by src/locus/eval/metrics.py.",
    "reassociation_performance": "SpatialMemoryGraph's re-association logic (memory/graph.py) isn't separately instrumented or scored by the eval harness.",
    "ttc_prediction_performance": "geometry/kinematics.py computes TTC, but the eval harness does not compare it against a ground-truth TTC.",
}


def main() -> None:
    intrinsics = synthetic.DEFAULT_INTRINSICS
    per_scenario = {}
    total_latency_s = {name: 0.0 for name in CONDITIONS}
    total_latency_frames = {name: 0 for name in CONDITIONS}

    for scenario_name, scenario_fn in SCENARIOS.items():
        frames, ground_truth = scenario_fn()
        questions = generate_questions(ground_truth, stride=1)

        condition_results = {}
        for cond_name, cond_spec in CONDITIONS.items():
            condition = cond_spec["make"](intrinsics)
            t0 = time.perf_counter()
            result = run_condition(condition, frames, ground_truth, questions)
            elapsed = time.perf_counter() - t0
            total_latency_s[cond_name] += elapsed
            total_latency_frames[cond_name] += len(frames)
            condition_results[cond_name] = {
                "in_view": result["in_view"],
                "out_of_view": result["out_of_view"],
                "wall_clock_s_for_scenario": round(elapsed, 4),
            }

        per_scenario[scenario_name] = {
            "n_frames": len(frames),
            "n_questions": len(questions),
            "n_objects": len(ground_truth.objects),
            "concepts": sorted({obj.concept for obj in ground_truth.objects}),
            "conditions": condition_results,
        }
        print(f"{scenario_name}: {len(frames)} frames, {len(questions)} questions -- done")

    # Aggregate across all scenarios, pooling questions (not averaging
    # per-scenario rates, since scenario question counts differ).
    aggregate = {}
    for cond_name in CONDITIONS:
        iv_n = sum(per_scenario[s]["conditions"][cond_name]["in_view"]["n_queries"] for s in SCENARIOS)
        oov_n = sum(per_scenario[s]["conditions"][cond_name]["out_of_view"]["n_queries"] for s in SCENARIOS)
        iv_answered = sum(
            round(per_scenario[s]["conditions"][cond_name]["in_view"]["answer_rate"] * per_scenario[s]["conditions"][cond_name]["in_view"]["n_queries"])
            for s in SCENARIOS
        )
        oov_answered = sum(
            round(per_scenario[s]["conditions"][cond_name]["out_of_view"]["answer_rate"] * per_scenario[s]["conditions"][cond_name]["out_of_view"]["n_queries"])
            for s in SCENARIOS
        )
        bearing_errs = [
            per_scenario[s]["conditions"][cond_name]["out_of_view"]["mean_bearing_error_hours"]
            for s in SCENARIOS
            if per_scenario[s]["conditions"][cond_name]["out_of_view"]["mean_bearing_error_hours"] is not None
        ]
        aggregate[cond_name] = {
            "label": CONDITIONS[cond_name]["label"],
            "description": CONDITIONS[cond_name]["description"],
            "in_view_answer_rate": round(iv_answered / iv_n, 4) if iv_n else None,
            "in_view_n_queries": iv_n,
            "out_of_view_answer_rate": round(oov_answered / oov_n, 4) if oov_n else None,
            "out_of_view_n_queries": oov_n,
            "out_of_view_mean_bearing_error_hours": round(sum(bearing_errs) / len(bearing_errs), 3) if bearing_errs else None,
            "mean_wall_clock_s_per_frame": round(total_latency_s[cond_name] / total_latency_frames[cond_name], 6),
            "note_on_latency": "Python-loop wall-clock time per synthetic frame through the condition's observe()+answer() path only -- not a deployed/real-time inference benchmark, no GPU model calls in this path (MockSegmenter is ground-truth-driven, no SAM/Spark/TTS). Useful only as a relative A/B/C/D comparison, not an absolute system latency figure.",
        }

    output = {
        "evaluation_protocol": {
            "description": "Synthetic ground-truth evaluation. All four conditions are run against the same MockSegmenter-driven detections (ground-truth-driven, not a real vision model) so that persistence behavior, not perception quality, is what's being compared. This isolates the one variable the ablation ladder is designed to test.",
            "why_synthetic": "Synthetic scenarios are the only data source with exact ground truth for visibility, bearing and range at every frame -- see docs/CLAUDE.md. This is NOT a SANPO/AEA real-world-ground-truth result.",
            "scenarios": list(SCENARIOS.keys()),
            "n_scenarios": len(SCENARIOS),
            "question_generation": "One question per (frame, object): 'where is <concept> right now', scored against exact ground-truth bearing/range, split by ground-truth visibility (in-view vs out-of-view) at query time.",
            "denominator_note": "in_view_n_queries / out_of_view_n_queries below are pooled question counts across all 5 scenarios, identical for every condition (same question set asked of every condition).",
        },
        "conditions": aggregate,
        "per_scenario": per_scenario,
        "metrics_not_implemented": NOT_IMPLEMENTED,
    }

    out_path = Path("dashboard/data/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Central result: A out-of-view answer rate = {aggregate['A_stateless']['out_of_view_answer_rate']}, "
          f"D out-of-view answer rate = {aggregate['D_mask']['out_of_view_answer_rate']}")


if __name__ == "__main__":
    main()
