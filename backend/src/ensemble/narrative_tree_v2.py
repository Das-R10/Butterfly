"""
src/ensemble/narrative_tree.py  (v2 — replaces v1)

Branching Narrative Tree with:
  - Zero-probability branch pruning
  - Best-path backtracking (highest cumulative probability leaf → root)
  - Dynamic bucket integration (uses score_and_map_dynamic per node)
  - Session memory threading (each path carries its history)

Tree mechanics:
  1. Run simulation on root signal → dynamic buckets + probability map
  2. Prune branches where probability == 0 or below MIN_PROBABILITY
  3. For each surviving branch, generate a narrative signal and re-simulate
  4. Recurse to MAX_DEPTH
  5. After tree is complete, backtrack from highest-probability leaf to root
     → this is the "most likely future" path
"""

from __future__ import annotations

import json
import os
import hashlib
import requests
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

MIN_PROBABILITY  = 0.05    # prune branches below this — zero branches always pruned
MAX_DEPTH        = 3
TOP_N_CHILDREN   = 3       # max children per node
SIGNAL_MAX_CHARS = 700

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

NARRATIVE_SIGNAL_PROMPT = """You are a news analyst. Write a 2-3 sentence intelligence briefing describing what the world looks like IF the following narrative has become dominant over the past 48 hours.

Write as factual present-tense. No "if" or "assuming". Be specific — name actors, locations, numbers where plausible.

Original signal:
{original_signal}

Narrative that has become dominant ({probability:.0%} probability):
{narrative_label}

Supporting personalities: {personalities}

Briefing (2-3 sentences, plain text):"""


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeNode:
    node_id:         str
    depth:           int
    bucket:          str
    label:           str
    probability:     float          # probability at this step (from parent map)
    cumulative_prob: float          # product of all step probabilities on path to here
    signal_text:     str
    simulation_result: dict = field(default_factory=dict)
    children:        list  = field(default_factory=list)
    parent_path:     list  = field(default_factory=list)  # list of (bucket, label) tuples
    timestamp:       str   = ""
    pruned:          bool  = False

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and not self.pruned

    @property
    def path_string(self) -> str:
        parts = [label for _, label in self.parent_path] + [self.label]
        return " → ".join(parts)

    def to_dict(self, slim: bool = False) -> dict:
        d = {
            "node_id":         self.node_id,
            "depth":           self.depth,
            "bucket":          self.bucket,
            "label":           self.label,
            "probability":     round(self.probability, 4),
            "cumulative_prob": round(self.cumulative_prob, 4),
            "path":            self.path_string,
            "signal_preview":  self.signal_text[:150],
            "is_leaf":         self.is_leaf,
            "n_children":      len(self.children),
        }
        if not slim:
            d["probability_map"] = self.simulation_result.get("clusters", [])
            d["dynamic_buckets"] = self.simulation_result.get("dynamic_buckets", [])
            d["dominant"]        = self.simulation_result.get("dominant_narrative", {})
        d["children"] = [c.to_dict(slim) for c in self.children]
        return d


# ── Tree ──────────────────────────────────────────────────────────────────────

class NarrativeTree:

    def __init__(
        self,
        run_simulation_fn,     # (signal, signal_id) → raw result dict
        score_fn=None,         # (result) → result with clusters; if None uses dynamic scorer
        max_depth:   int   = MAX_DEPTH,
        top_n:       int   = TOP_N_CHILDREN,
        min_prob:    float = MIN_PROBABILITY,
        interval_label: str = "48h",
    ):
        self.run_sim   = run_simulation_fn
        self.score_fn  = score_fn or self._default_score
        self.max_depth = max_depth
        self.top_n     = top_n
        self.min_prob  = min_prob
        self.interval  = interval_label
        self.nodes_run = 0

    def _default_score(self, result: dict) -> dict:
        from dynamic_bucket_scorer import score_and_map_dynamic
        return score_and_map_dynamic(result)

    def build(self, root_signal: str, root_result: dict = None) -> NarrativeNode:
        logger.info(f"Building narrative tree depth={self.max_depth} top_n={self.top_n} min_prob={self.min_prob}")

        root_id = f"root_{hashlib.md5(root_signal.encode()).hexdigest()[:8]}"

        if root_result is None:
            logger.info("Running root simulation...")
            root_result = self.run_sim(signal=root_signal, signal_id=root_id)
            root_result = self.score_fn(root_result)

        root = NarrativeNode(
            node_id=root_id, depth=0, bucket="root", label="Signal t=0",
            probability=1.0, cumulative_prob=1.0,
            signal_text=root_signal, simulation_result=root_result,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        self._expand(root, [], 1.0)
        logger.info(f"Tree built — {self.nodes_run} simulations, {count_nodes(root)} total nodes")
        return root

    def _expand(self, node: NarrativeNode, parent_path: list, parent_cum: float):
        if node.depth >= self.max_depth:
            return

        clusters = node.simulation_result.get("clusters", [])
        if not clusters:
            return

        # PRUNE: remove zero-prob and below-threshold branches
        candidates = [
            c for c in clusters
            if c.get("probability", 0) > 0 and c.get("probability", 0) >= self.min_prob
        ]
        candidates = sorted(candidates, key=lambda x: -x["probability"])[:self.top_n]

        if not candidates:
            logger.debug(f"Node {node.node_id} — all branches pruned (below {self.min_prob})")
            return

        logger.info(f"Expanding depth={node.depth} '{node.label}' → {len(candidates)} children")

        new_path = parent_path + [(node.bucket, node.label)] if node.bucket != "root" else []

        for cluster in candidates:
            bk    = cluster.get("bucket", "unknown")
            label = cluster.get("label", bk)
            prob  = cluster.get("probability", 0.0)
            cum   = parent_cum * prob

            # Generate child signal
            child_signal = self._narrative_to_signal(
                original_signal=node.signal_text,
                label=label, prob=prob,
                personalities=cluster.get("personalities", []),
                path_context=self._path_context(new_path),
            )

            child_id = f"d{node.depth+1}_{bk[:8]}_{hashlib.md5(child_signal.encode()).hexdigest()[:6]}"

            logger.info(f"  Running child: '{label}' ({prob:.0%}) cumul={cum:.1%}")
            self.nodes_run += 1

            try:
                child_result = self.run_sim(signal=child_signal, signal_id=child_id)
                child_result = self.score_fn(child_result)
            except Exception as e:
                logger.error(f"Child sim failed: {e}")
                continue

            child = NarrativeNode(
                node_id=child_id, depth=node.depth+1,
                bucket=bk, label=label,
                probability=prob, cumulative_prob=cum,
                signal_text=child_signal, simulation_result=child_result,
                parent_path=new_path,
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            node.children.append(child)
            self._expand(child, new_path + [(bk, label)], cum)

    def _narrative_to_signal(
        self, original_signal: str, label: str, prob: float,
        personalities: list, path_context: str,
    ) -> str:
        prompt = NARRATIVE_SIGNAL_PROMPT.format(
            original_signal=original_signal[:300],
            probability=prob, narrative_label=label,
            personalities=", ".join(personalities[:4]),
        )
        if path_context:
            prompt = f"[Path context: {path_context}]\n\n" + prompt

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.3, "num_predict": 200}},
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                if text and len(text) > 40:
                    return text[:SIGNAL_MAX_CHARS]
        except Exception as e:
            logger.warning(f"Signal generation failed: {e}")
        return f"{label} scenario: {original_signal[:200]}"

    def _path_context(self, path: list) -> str:
        if not path:
            return ""
        steps = " → ".join(f"{label}" for _, label in path)
        return f"Prior narrative path: {steps}"


# ── Best-path backtracking ────────────────────────────────────────────────────

def find_best_path(root: NarrativeNode) -> list[NarrativeNode]:
    """
    Find the highest cumulative probability path through the tree.
    Strategy: walk to every leaf, collect cumulative_prob, backtrack
    from the best leaf to root by following parent pointers.

    Returns list of nodes from root to best leaf.
    """
    best_leaf  = None
    best_prob  = -1.0

    def walk_leaves(node: NarrativeNode):
        nonlocal best_leaf, best_prob
        if node.is_leaf:
            if node.cumulative_prob > best_prob:
                best_prob = node.cumulative_prob
                best_leaf = node
            return
        for child in node.children:
            walk_leaves(child)

    walk_leaves(root)

    if best_leaf is None:
        return [root]

    # Backtrack from best leaf to root using parent_path metadata
    # Reconstruct by re-walking the tree following the path
    path_labels = [label for _, label in best_leaf.parent_path] + [best_leaf.label]

    def trace_path(node: NarrativeNode, remaining: list) -> list[NarrativeNode]:
        if not remaining:
            return [node]
        target = remaining[0]
        for child in node.children:
            if child.label == target:
                result = trace_path(child, remaining[1:])
                if result:
                    return [node] + result
        return [node]

    return trace_path(root, path_labels[1:]) if len(path_labels) > 1 else [root]


def get_all_leaves(root: NarrativeNode) -> list[NarrativeNode]:
    """Return all leaf nodes sorted by cumulative probability descending."""
    leaves = []
    def walk(node):
        if node.is_leaf:
            leaves.append(node)
        for child in node.children:
            walk(child)
    walk(root)
    return sorted(leaves, key=lambda n: -n.cumulative_prob)


def count_nodes(root: NarrativeNode) -> int:
    return 1 + sum(count_nodes(c) for c in root.children)


def compute_convergence(root: NarrativeNode) -> dict:
    """Does the tree converge on one narrative regardless of path?"""
    leaves = get_all_leaves(root)
    if not leaves:
        return {}
    from collections import Counter
    counts = Counter(l.bucket for l in leaves)
    top    = counts.most_common(1)[0]
    conv   = top[1] / len(leaves)
    return {
        "convergence":         round(conv, 4),
        "dominant_endpoint":   top[0],
        "endpoint_label":      leaves[0].label if leaves else "",
        "n_leaves":            len(leaves),
        "interpretation":      (
            "self-reinforcing — narrative dominates all paths"  if conv > 0.65 else
            "partially convergent — one narrative leads"        if conv > 0.4  else
            "chaotic — path-dependent, no dominant endpoint"
        ),
    }


def summarize_tree(root: NarrativeNode, interval: str = "48h") -> str:
    best_path  = find_best_path(root)
    leaves     = get_all_leaves(root)
    convergence = compute_convergence(root)

    lines = [
        "=" * 72,
        "  BUTTERFLY ALGORITHM — NARRATIVE TREE",
        "=" * 72,
        f"  Root: {root.signal_text[:90]}...",
        f"  Depth: {root.depth} → {max(l.depth for l in leaves) if leaves else 0} cycles",
        f"  Interval: {interval} per cycle",
        f"  Nodes: {count_nodes(root)} total, {len(leaves)} leaves",
        f"  Convergence: {convergence.get('convergence','?')} — {convergence.get('interpretation','')}",
        "",
        "─" * 72,
        "  BEST PATH (highest cumulative probability)",
        "─" * 72,
    ]

    for node in best_path[1:]:
        lines.append(
            f"  t{node.depth} [{interval}×{node.depth}]: {node.label}"
            f" — step {node.probability:.0%}, cumul {node.cumulative_prob:.1%}"
        )
        lines.append(f"    {node.signal_text[:100]}...")
        lines.append("")

    lines += ["─" * 72, "  ALL LEAVES (sorted by cumulative probability)", "─" * 72]
    for leaf in leaves[:12]:
        lines.append(
            f"  {leaf.cumulative_prob:.1%}  {leaf.path_string}"
        )

    lines += ["", "=" * 72, f"  Generated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC", "=" * 72]
    return "\n".join(lines)


def save_tree(root: NarrativeNode, path: str = None, slim: bool = False) -> str:
    path = path or f"data/outputs/tree_{root.node_id}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(root.to_dict(slim=slim), f, indent=2)
    logger.info(f"Tree saved → {path}")
    return path
