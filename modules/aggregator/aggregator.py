from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from .agg_utils import (
    ExpertPacket, Claim, ClaimCluster,
    load_embedder, extract_assumptions, estimate_uncertainty,
    calibrate_reliabilities, extract_claims, cluster_claims,
    claim_truth_scores, contradiction_pairs, weighted_mmr_select,
    compose_draft, build_editor_prompt
)
import numpy as np

class Aggregator:
    @dataclass
    class Config:
        alpha_softmax: float = 3.0
        sim_thresh: float = 0.82
        selection_budget: int = 24
        lambda_redundancy: float = 0.6
        use_nli: bool = False
        nli_topK: int = 40
        sections: List[str] = None
        domain_defaults: List[str] = None

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Aggregator.Config()
        if self.cfg.sections is None:
            self.cfg.sections = ["Assumptions", "Method", "Results", "Caveats"]
        _ = load_embedder()  # warm-up

    @staticmethod
    def _normalize_experts(experts: Any) -> List[ExpertPacket]:
        packets: List[ExpertPacket] = []
        if isinstance(experts, dict):
            for eid, obj in experts.items():
                if isinstance(obj, str):
                    packets.append(ExpertPacket(expert_id=eid, answer=obj))
                else:
                    packets.append(ExpertPacket(
                        expert_id=eid,
                        answer=obj.get("answer", ""),
                        router_confidence=float(obj.get("router_confidence", 0.5)),
                        scope=obj.get("scope", ""),
                        evidence=obj.get("evidence", []) or [],
                    ))
        elif isinstance(experts, list):
            for obj in experts:
                packets.append(ExpertPacket(
                    expert_id=obj.get("expert_id") or obj.get("id") or f"E{len(packets)}",
                    answer=obj.get("answer", ""),
                    router_confidence=float(obj.get("router_confidence", 0.5)),
                    scope=obj.get("scope", ""),
                    evidence=obj.get("evidence", []) or [],
                ))
        else:
            raise ValueError("experts must be a dict or list of dicts")
        return packets

    def _embed_texts(self, texts: List[str]):
        EMB = load_embedder()
        return EMB.encode(texts, normalize_embeddings=True)

    def run(self, question: str, experts: Any) -> Dict[str, Any]:
        packets = self._normalize_experts(experts)

        # 1) Fill assumptions + uncertainty
        answers = [p.answer for p in packets]
        A = self._embed_texts(answers) if answers else np.zeros((0, 384))
        for i, p in enumerate(packets):
            p.assumptions = extract_assumptions(p.answer, domain_defaults=self.cfg.domain_defaults)
            others = [A[j] for j in range(len(packets)) if j != i]
            this_emb = A[i] if len(answers) > i else (np.zeros_like(others[0]) if others else np.zeros((384,)))
            p.uncertainty_self = estimate_uncertainty(
                router_rho=float(p.router_confidence),
                answer=p.answer, this_emb=this_emb, others_embs=others
            )

        # 2) Reliability using router + uncertainty
        reliab = calibrate_reliabilities(packets, alpha=self.cfg.alpha_softmax)

        # 3) Claims -> clusters -> scores
        claims = [cl for p in packets for cl in extract_claims(p)]
        clusters = cluster_claims(claims, sim_thresh=self.cfg.sim_thresh)
        scores = claim_truth_scores(clusters, reliab)

        # 4) Optional contradiction pairs
        forbids = set()
        if self.cfg.use_nli and clusters:
            forbids = contradiction_pairs(clusters, scores, topK=self.cfg.nli_topK)

        # 5) Selection (MMR + forbid)
        clusters_sorted = sorted(clusters, key=lambda c: -scores.get(c.cluster_id, 0.0))
        selected = weighted_mmr_select(
            clusters_sorted, scores,
            budget=self.cfg.selection_budget,
            lambda_red=self.cfg.lambda_redundancy,
            forbid_pairs=forbids
        )

        # 6) Render
        draft = compose_draft(question, self.cfg.sections, selected, scores)
        editor_prompt = build_editor_prompt(question, selected, scores, sections=self.cfg.sections)

        # 7) Audit bundle
        selected_summary = [{
            "cluster_id": c.cluster_id,
            "score": round(float(scores.get(c.cluster_id, 0.0)), 3),
            "experts": list({m.expert_id for m in c.members}),
            "claims": [m.text for m in c.members]
        } for c in selected]

        return {
            "draft": draft,
            "editor_prompt": editor_prompt,
            "selected_claims": selected_summary,
            "packets": packets,
            "forbidden_pairs": list(forbids),
        }

# -------- CLI smoke test --------
if __name__ == "__main__":
    example_experts = {
        "PhysicsKinematics": {
            "answer": (
                "- Assume air resistance is negligible.\n"
                "- Range is v^2 * sin(2θ) / g.\n"
                "If the launch angle is 45°, the range is maximized."
            ),
            "router_confidence": 0.72
        },
        "NumericsVerifier": {
            "answer": (
                "Given g = 9.81 m/s^2. Approximate v = 20 m/s, θ = 30°.\n"
                "Then range ≈ (20^2 * sin(60°)) / 9.81 ≈ 35.3 m."
            ),
            "router_confidence": 0.58
        },
        "EdgeCaseCaveats": {
            "answer": (
                "This assumes flat ground and no wind.\n"
                "Results may vary if terrain is uneven or g differs."
            ),
            "router_confidence": 0.40
        }
    }
    question = "Compute the projectile range and list the assumptions clearly."
    agg = Aggregator(Aggregator.Config(
        use_nli=False, selection_budget=12, lambda_redundancy=0.6
    ))
    result = agg.run(question, example_experts)

    print("\n===== RAW DRAFT =====\n")
    print(result["draft"])
    print("\n===== EDITOR PROMPT =====\n")
    print(result["editor_prompt"])
    print("\n===== SELECTED CLAIMS =====\n")
    for c in result["selected_claims"]:
        print(c)
    print("\n===== EXPERT PACKETS (enriched) =====\n")
    for p in result["packets"]:
        print(vars(p))
