from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re, numpy as np

# -------------- Embeddings / NLI loaders --------------
_EMB = None
def load_embedder():
    """Lazy-load a light sentence embedding model."""
    global _EMB
    if _EMB is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(
                "Install sentence-transformers (pip install sentence-transformers) to use Aggregator."
            ) from e
    return _EMB

_NLI = None
_TOK = None
def load_nli():
    """Lazy-load an MNLI model; only if contradiction checks are enabled."""
    global _NLI, _TOK
    if _NLI is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "roberta-large-mnli"
        _TOK = AutoTokenizer.from_pretrained(name)
        _NLI = AutoModelForSequenceClassification.from_pretrained(name)
    return _NLI, _TOK

# -------------- Data models --------------
@dataclass
class ExpertPacket:
    expert_id: str
    answer: str
    scope: str = ""
    router_confidence: float = 0.5       # [0,1]
    uncertainty_self: float = 0.5        # filled by estimator
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    numerics: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)

@dataclass
class Claim:
    claim_id: str
    text: str
    expert_id: str
    evidence_refs: List[str]
    line_no: int

@dataclass
class ClaimCluster:
    cluster_id: str
    members: List[Claim]
    centroid: np.ndarray
    supports: Dict[str, float] = field(default_factory=dict)

# -------------- Text utils --------------
HEDGE_TERMS = set("""
likely maybe may might roughly around approximately estimate could should appears suggest suggestive
presumably probably possibly near about typically generally seemingly arguably
""".split())

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]

# -------------- Assumption extraction --------------
_ASSUMP_CUES = [
    r"\\bassum(?:e|ing)\\b.+", r"\\bgiven\\b.+", r"\\bprovided that\\b.+", r"\\bin the absence of\\b.+",
    r"\\bneglect(?:ing)?\\b.+", r"\\bignoring\\b.+", r"\\b(treat|model)\\b.+\\bas\\b.+",
    r"\\bhold(?:ing)? .+ fixed\\b", r"\\bapprox(?:imate|imation|imately)\\b.+",
]
_COND = r"\\b(if|when|unless)\\b.+"

def extract_assumptions(answer: str, domain_defaults: Optional[List[str]] = None, cap: int = 8) -> List[str]:
    assumptions = set()
    for pat in _ASSUMP_CUES:
        rx = re.compile(pat, re.I)
        for m in rx.finditer(answer):
            assumptions.add(m.group().strip(" ."))

    cond_rx = re.compile(_COND, re.I)
    for s in split_sentences(answer):
        if cond_rx.match(s):
            assumptions.add(s)

    for s in split_sentences(answer):
        toks = re.findall(r"[a-zA-Z']+", s.lower())
        if not toks: 
            continue
        hedge_ratio = sum(t in HEDGE_TERMS for t in toks) / max(1, len(toks))
        if hedge_ratio >= 0.06 and 30 <= len(s) <= 160:
            assumptions.add(s)

    if not assumptions and domain_defaults:
        assumptions.update(domain_defaults)

    out, seen = [], set()
    for a in assumptions:
        k = a.lower()
        if k not in seen:
            seen.add(k); out.append(a)
    return out[:cap]

# -------------- Uncertainty estimation --------------
def _evidence_score(answer: str) -> float:
    has_url = bool(re.search(r"https?://", answer))
    has_num = bool(re.search(r"\\d", answer))
    has_eq  = any(sym in answer for sym in ["=", "≈", "≤", "≥", "+", "-", "*", "/"])
    return min(1.0, 0.5*has_url + 0.3*has_num + 0.2*has_eq)

def _hedge_score(answer: str) -> float:
    toks = re.findall(r"[a-zA-Z']+", answer.lower())
    if not toks: return 0.0
    h = sum(t in HEDGE_TERMS for t in toks) / len(toks)
    return min(1.0, 3.0*h)

def _verification_score(_answer: str) -> float:
    return 0.5  # hook for unit/math checks

def _agreement_score(this_emb: np.ndarray, others_embs: List[np.ndarray]) -> float:
    if not others_embs: return 0.5
    sims = [float(this_emb @ v) for v in others_embs]
    sims = sorted(sims, reverse=True)[:2]
    return float(np.mean(sims) * 0.5 + 0.5)

def estimate_uncertainty(router_rho: float, answer: str, this_emb: np.ndarray, others_embs: List[np.ndarray]) -> float:
    e = _evidence_score(answer)
    h = _hedge_score(answer)
    v = _verification_score(answer)
    a = _agreement_score(this_emb, others_embs)
    w_r, w_e, w_v, w_a, w_h = 0.40, 0.20, 0.10, 0.20, 0.10
    quality = w_r*router_rho + w_e*e + w_v*v + w_a*a + w_h*(1.0 - h)
    quality = max(0.0, min(1.0, quality))
    return float(1.0 - quality)

# -------------- Reliability calibration --------------
def softmax_weights(scores: List[float], alpha: float = 3.0) -> List[float]:
    x = np.array(scores, dtype=float) * alpha
    e = np.exp(x - x.max())
    w = e / (e.sum() + 1e-12)
    return w.tolist()

def calibrate_reliabilities(packets: List[ExpertPacket], alpha: float = 3.0) -> Dict[str, float]:
    w = softmax_weights([p.router_confidence for p in packets], alpha=alpha)
    reliab = {}
    for p, wi in zip(packets, w):
        r = wi * (1.0 - p.uncertainty_self)   # shrink by uncertainty
        reliab[p.expert_id] = max(0.02, min(0.98, float(r)))
    return reliab

# -------------- Claim extraction & clustering --------------
def extract_claims(p: ExpertPacket) -> List[Claim]:
    lines = [ln.strip(" -•\\t") for ln in p.answer.splitlines() if ln.strip()] or split_sentences(p.answer)
    claims: List[Claim] = []
    for i, ln in enumerate(lines, 1):
        if len(ln) < 3: 
            continue
        claims.append(Claim(
            claim_id=f"{p.expert_id}:{i}",
            text=ln,
            expert_id=p.expert_id,
            evidence_refs=p.evidence[:3],
            line_no=i
        ))
    return claims

def cluster_claims(claims: List[Claim], sim_thresh: float = 0.82) -> List[ClaimCluster]:
    if not claims: return []
    EMB = load_embedder()
    texts = [c.text for c in claims]
    X = EMB.encode(texts, normalize_embeddings=True)
    clusters, used = [], set()

    for i in range(len(claims)):
        if i in used: continue
        members = [i]
        sims = X @ X[i]
        for j in range(i+1, len(claims)):
            if j in used: continue
            if sims[j] >= sim_thresh:
                members.append(j); used.add(j)
        used.add(i)
        mem_claims = [claims[k] for k in members]
        centroid = X[members].mean(axis=0)
        clusters.append(ClaimCluster(
            cluster_id=f"C{len(clusters)}", members=mem_claims, centroid=centroid
        ))
    return clusters

# -------------- Claim truth scoring --------------
def claim_truth_scores(clusters: List[ClaimCluster], reliab: Dict[str, float]) -> Dict[str, float]:
    scores = {}
    for cl in clusters:
        rhos = [reliab.get(m.expert_id, 0.1) for m in cl.members]
        if not rhos:
            scores[cl.cluster_id] = 0.0
        else:
            one_minus = np.prod([1.0 - r for r in rhos])  # prob none reliable
            scores[cl.cluster_id] = float(1.0 - one_minus)
        cl.supports = {m.expert_id: reliab.get(m.expert_id, 0.1) for m in cl.members}
    return scores

# -------------- NLI contradiction (optional) --------------
def nli_relation(a: str, b: str) -> str:
    NLI, TOK = load_nli()
    import torch
    inputs = TOK(a, b, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = NLI(**inputs).logits.softmax(dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))  # 0: contradiction, 1: neutral, 2: entailment
    return ["contradicts", "neutral", "entails"][idx]

def contradiction_pairs(clusters: List[ClaimCluster], scores: Dict[str, float], topK: int = 40) -> set[tuple[str,str]]:
    top = sorted(clusters, key=lambda c: -scores[c.cluster_id])[:min(topK, len(clusters))]
    pairs = set()
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            a = " ".join(m.text for m in top[i].members[:2])
            b = " ".join(m.text for m in top[j].members[:2])
            if nli_relation(a, b) == "contradicts":
                pairs.add((top[i].cluster_id, top[j].cluster_id))
    return pairs

# -------------- Selection (Weighted MMR) --------------
def weighted_mmr_select(clusters: List[ClaimCluster],
                        scores: Dict[str, float],
                        budget: int = 20,
                        lambda_red: float = 0.6,
                        forbid_pairs: Optional[set[tuple[str,str]]] = None) -> List[ClaimCluster]:
    forbid_pairs = forbid_pairs or set()
    selected, selected_vecs = [], []

    def _conflicts(cid: str) -> bool:
        for s in selected:
            if (cid, s.cluster_id) in forbid_pairs or (s.cluster_id, cid) in forbid_pairs:
                return True
        return False

    for _ in range(min(budget, len(clusters))):
        best, best_sc = None, -1e9
        for c in clusters:
            if c in selected or _conflicts(c.cluster_id): 
                continue
            base = scores.get(c.cluster_id, 0.0)
            red = max([float(c.centroid @ v) for v in selected_vecs], default=0.0)
            mmr = base - lambda_red * red
            if mmr > best_sc:
                best_sc, best = mmr, c
        if best is None: break
        selected.append(best); selected_vecs.append(best.centroid)
    return selected

# -------------- Section classification & rendering --------------
ASSUME_CUES = (
    r"\\bassum(?:e|ing)\\b", r"\\bgiven\\b", r"\\bprovided that\\b", r"\\bif\\b", r"\\bwhen\\b", r"\\bunless\\b",
    r"\\bneglect(?:ing)?\\b", r"\\bignoring\\b", r"\\b(treat|model)\\b.+\\bas\\b", r"\\bhold(?:ing)? .+ fixed\\b"
)
METHOD_CUES = (r"\\busing\\b", r"\\bwe use\\b", r"\\bcompute\\b", r"\\bformula\\b", r"\\bequation\\b", r"\\bderive\\b")
RESULT_CUES = (r"\\btherefore\\b", r"\\bthus\\b", r"\\bso\\b", r"\\bresult\\b", r"\\bequals\\b", r"≈", r"=", r"\\bthe (answer|range|value) is\\b")
CAVEAT_CUES = (r"\\bmay vary\\b", r"\\bedge case\\b", r"\\bif not\\b", r"\\bunless\\b", r"\\buncertain\\b", r"\\blimit(?:ation)?\\b")
_num_like = re.compile(r"\\d|\\bdeg\\b|°|m/s|ms-?2|m\\b|kg\\b|%|\\bETA\\b", re.I)

def classify_section(text: str) -> Optional[str]:
    t = text.strip().lower()
    if any(re.search(p, t) for p in ASSUME_CUES): return "Assumptions"
    if (any(re.search(p, t) for p in RESULT_CUES) and _num_like.search(t)) or t.startswith("then "): return "Results"
    if any(re.search(p, t) for p in CAVEAT_CUES): return "Caveats"
    if any(re.search(p, t) for p in METHOD_CUES) or ("sin(" in t or "cos(" in t or "^" in t): return "Method"
    return None

def compose_draft(question: str,
                  sections: List[str],
                  selected: List[ClaimCluster],
                  scores: Dict[str, float]) -> str:
    EMB = load_embedder()
    sec_vecs = EMB.encode(sections, normalize_embeddings=True)
    buckets = {s: [] for s in sections}

    def best_section_for_cluster(c: ClaimCluster) -> str:
        votes = [sec for m in c.members if (sec := classify_section(m.text))]
        if votes:
            return max(set(votes), key=votes.count)
        sims = [float(c.centroid @ v) for v in sec_vecs]
        return sections[int(np.argmax(sims))]

    placed = []
    for c in selected:
        sec = best_section_for_cluster(c)
        uniq = []
        for m in c.members:
            line = f"- {m.text} [{m.expert_id}] (score={scores[c.cluster_id]:.2f})"
            if line not in uniq: uniq.append(line)
        buckets[sec].extend(uniq)
        placed.append((c, sec))

    if not buckets.get("Results"):
        def resultiness(txt: str) -> float:
            r = 0.0
            if _num_like.search(txt): r += 0.6
            if any(k in txt.lower() for k in ["≈", "=", "therefore", "thus", "result", " is "]): r += 0.4
            return r
        candidates = []
        for c, sec in placed:
            if sec == "Method":
                for m in c.members:
                    candidates.append((resultiness(m.text) * scores[c.cluster_id], c, m))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, c_best, m_best = candidates[0]
            line = f"- {m_best.text} [{m_best.expert_id}] (score={scores[c_best.cluster_id]:.2f})"
            if line in buckets["Method"]:
                buckets["Method"].remove(line)
            buckets["Results"].append(line)

    draft = [f"# Final Answer\\n\\n**Question:** {question}\\n"]
    for sec in sections:
        draft.append(f"## {sec}\\n")
        draft.extend(buckets[sec] or ["- (none)"])
        draft.append("")
    conf = np.mean(sorted([scores.get(c.cluster_id, 0.0) for c, _ in placed], reverse=True)[:max(1, len(placed)//2)])
    draft.append("## Confidence & Caveats")
    draft.append(f"Overall confidence: {conf:.2f}")
    draft.append("- Selected a non-redundant, high-support subset of claims.\\n")
    return "\\n".join(draft)

def build_editor_prompt(question: str,
                        selected: List[ClaimCluster],
                        scores: Dict[str, float],
                        sections: Optional[List[str]] = None) -> str:
    sections = sections or ["Assumptions", "Method", "Results", "Caveats"]
    EMB = load_embedder()
    sec_vecs = EMB.encode(sections, normalize_embeddings=True)
    buckets = {s: [] for s in sections}

    for c in selected:
        votes = [sec for m in c.members if (sec := classify_section(m.text))]
        if votes:
            sec = max(set(votes), key=votes.count)
        else:
            sims = [float(c.centroid @ v) for v in sec_vecs]
            sec = sections[int(np.argmax(sims))]

        for m in c.members:
            line = f"- {m.text} [{m.expert_id}] (score={scores[c.cluster_id]:.2f})"
            if line not in buckets[sec]:
                buckets[sec].append(line)

    # Guarantee non-empty Results for the editor, mirroring compose_draft behavior
    if not buckets["Results"] and buckets["Method"]:
        # move the last (likely the most procedural) method line to Results as a fallback
        buckets["Results"].append(buckets["Method"].pop())

    lines = []
    lines.append("You are an editor. Convert the provided consistent, high-confidence claims into a coherent final answer.")
    lines.append("Preserve factual details; do not introduce new claims.")
    lines.append(f"\\nQuestion:\\n{question}\\n")
    lines.append("Selected Claims:")
    for sec in sections:
        lines.append(f"## {sec}")
        if buckets[sec]:
            lines.extend(buckets[sec])
        else:
            lines.append("- (none)")
        lines.append("")
    lines.append("Requirements:\\n1) Merge into clear prose.\\n2) Keep key attributions in brackets.\\n3) End with 'Confidence & Caveats'.")
    return "\\n".join(lines)
