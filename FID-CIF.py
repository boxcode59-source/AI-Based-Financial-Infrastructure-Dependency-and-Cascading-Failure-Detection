"""
FID-CIF: Financial Infrastructure Dependency and Cascade Intelligence Framework
================================================================================
Single-file, working implementation of the three-component architecture:

  MFTDG   - Multi-Layer Financial-Technology Dependency Graph construction
            (evidence extraction -> confidence fusion -> evidentiary banding
             -> temporal multi-layer graph), Sections 3.4-3.6, Eqs. (1)-(4)

  CI-FCPN - Causal Infrastructure-to-Financial Cascade Propagation Network
            (do()-intervention, multi-stage health-state recursion, financial
             readout), Section 3.8, Eqs. (10)-(14)

  ADCRAE  - Adaptive Dependency Concentration and Resilience Assessment Engine
            (concentration scoring, critical-combination search, capacitated
             recovery assignment), Sections 3.7 & 3.9, Eqs. (5)-(9), (15)-(18)

  + Section 3.10 severity/composite index (Eqs. 19-20) and the Section 3.11
    Algorithm 1 end-to-end orchestration.

Everything not claimed as a contribution in the paper (record linkage,
NER/RE, BFS, Jaccard, logistic regression, greedy submodular maximization,
the Hungarian algorithm, semi-Markov recovery) is implemented with standard,
off-the-shelf techniques. The novel equations (confidence fusion, DC_p,
the kappa/rho-coupled propagation recursion, the critical-combination gap,
and FICSI) are implemented as given in the paper.

No real institution-to-vendor dataset exists (this is exactly the
partial-observability problem P1 the paper describes), so this script ships
a synthetic-ecosystem generator that plays the role of the harmonized
FAICID corpus (Section 3.2-3.3) and produces a runnable end-to-end demo.
Swap `generate_synthetic_ecosystem()` for real ingestion code and everything
downstream is unchanged.

Dependencies: numpy, scipy, scikit-learn, networkx (all standard).
Run directly:  python fid_cif.py
"""

from __future__ import annotations

import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression


# ======================================================================
# Global configuration (thresholds / weights named in the paper)
# ======================================================================

class Config:
    # Section 3.5 banding thresholds (Eq. 3)
    THETA_PROB = 0.75   # >= => observed (if source is primary) or promoted band
    THETA_CONF = 0.45   # >= => inferred band; below => synthetic/unsupported

    # Section 3.4 entity-resolution match threshold (Eq. 1)
    THETA_ER = 0.88

    # Section 3.7 concentration-score fusion weights (Eq. 9): lambda1..4
    LAMBDA = dict(l1=0.35, l2=0.25, l3=0.20, l4=0.20)
    K_HOP = 2  # BFS depth used for Conc_p / Overlap_p

    # Section 3.8 layer-pair transmission coefficients kappa(l) (Eq. 10),
    # calibrated (here: hand-set as stand-ins for MLE-fit-from-incidents)
    KAPPA = {
        ("cloud", "ai_service"): 0.85,
        ("ai_service", "application"): 0.80,
        ("data_provider", "ai_service"): 0.55,
        ("identity", "application"): 0.65,
        ("cloud", "application"): 0.70,
        ("payment", "application"): 0.75,
        ("application", "business_process"): 0.90,
        ("business_process", "financial_service"): 0.95,
        ("cloud", "identity"): 0.60,
        ("cloud", "payment"): 0.60,
    }
    KAPPA_DEFAULT = 0.5

    # Section 3.9 recovery / semi-Markov state machine
    RECOVERY_STATES = ("Primary", "Backup", "Degraded", "Manual")

    # Section 3.9 greedy critical-combination search
    MAX_COMBO_SIZE = 3  # k in Eq. (15)-(16)

    # Section 3.10 FICSI fusion weights gamma1..3 (Eq. 20)
    GAMMA = dict(g1=0.40, g2=0.35, g3=0.25)
    H_MIN = 0.5  # health threshold defining "failed" in Eq. (19)

    SEED = 42


# ======================================================================
# Section 3.4 — Entity Resolution (established record linkage, used as-is)
# ======================================================================

def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Standard Jaro-Winkler string similarity (Winkler, 1990)."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    match_dist = max(match_dist, 0)
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        lo, hi = max(0, i - match_dist), min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions, k = 0, 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions) / matches) / 3.0
    prefix = 0
    for a, b in zip(s1[:max_prefix], s2[:max_prefix]):
        if a != b:
            break
        prefix += 1
    return jaro + prefix * p * (1 - jaro)


class UnionFind:
    """Disjoint-set structure used to cluster matched entity mentions."""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def resolve_entities(mentions: List[str], legal_ids: Dict[str, Optional[str]],
                      theta_er: float = Config.THETA_ER) -> Dict[str, str]:
    """
    Section 3.4, Eq. (1): canopy-blocked, Fellegi-Sunter-style probabilistic
    record linkage collapsed here to its practical form -- exact match on a
    legal-entity identifier where available, else Jaro-Winkler name/domain
    similarity above theta_er -- clustered via Union-Find. Canopy blocking
    is applied on the first 3 characters as a cheap prefix key.
    """
    uf = UnionFind(mentions)
    # exact legal-id matches
    by_id: Dict[str, List[str]] = {}
    for m in mentions:
        lid = legal_ids.get(m)
        if lid:
            by_id.setdefault(lid, []).append(m)
    for group in by_id.values():
        for a, b in zip(group, group[1:]):
            uf.union(a, b)
    # canopy blocking on name prefix, then pairwise Jaro-Winkler
    blocks: Dict[str, List[str]] = {}
    for m in mentions:
        blocks.setdefault(m[:3].lower(), []).append(m)
    for block in blocks.values():
        for a, b in itertools.combinations(block, 2):
            if jaro_winkler(a, b) >= theta_er:
                uf.union(a, b)
    canon = {}
    for m in mentions:
        canon[m] = uf.find(m)
    return canon  # mention -> canonical id


# ======================================================================
# Section 3.5 — Confidence Estimation (Eq. 2-3): the paper's own equation
# ======================================================================

@dataclass
class CandidateEdge:
    src: str
    dst: str
    layer_pair: Tuple[str, str]
    phi_str: float   # evidence-statement strength (1.0 = explicit statement)
    phi_rel: float   # source reliability
    phi_cor: float   # normalized cross-source corroboration count
    is_primary_source: bool
    synthetic_target_stat: Optional[float] = None  # rho_agg, if synthetic


class ConfidenceEstimator:
    """
    Eq. (2): c_ij(t) = sigma(w1*phi_str + w2*phi_rel + w3*phi_cor)
    Weights w = (w1,w2,w3) fit by logistic regression against a small
    held-out, manually verified relationship set (Section 3.5 / Tier 2).
    """

    def __init__(self):
        self.model = LogisticRegression(C=10.0)
        self._fitted = False

    def fit(self, verified_features: np.ndarray, verified_labels: np.ndarray):
        self.model.fit(verified_features, verified_labels)
        self._fitted = True

    def score(self, cand: CandidateEdge) -> float:
        x = np.array([[cand.phi_str, cand.phi_rel, cand.phi_cor]])
        if self._fitted:
            return float(self.model.predict_proba(x)[0, 1])
        # fallback closed-form sigmoid with sensible hand weights
        w = np.array([2.2, 1.4, 1.0])
        z = float(x @ w) - 2.0
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def band(confidence: float, is_primary_source: bool,
              is_synthetic_request: bool,
              theta_prob=Config.THETA_PROB, theta_conf=Config.THETA_CONF) -> str:
        """Eq. (3): observed / inferred / synthetic evidentiary banding."""
        if is_synthetic_request:
            return "synthetic"
        if confidence >= theta_prob and is_primary_source:
            return "observed"
        if confidence >= theta_conf:
            return "inferred"
        return "synthetic"


# ======================================================================
# Section 3.6 — MFTDG: temporal multi-layer dependency graph (Eq. 4)
# ======================================================================

class MFTDG:
    """
    Wraps a networkx MultiDiGraph. Nodes are tagged by layer (financial
    institution, ai_service, cloud, data_provider, identity, payment,
    application, business_process, financial_service). Edges carry
    confidence, evidentiary band, and a temporal validity window
    [t_start, t_end]; nothing is hard-thresholded away (Section 3.6).
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()

    def add_entity(self, node_id: str, layer: str, **attrs):
        self.G.add_node(node_id, layer=layer, **attrs)

    def add_dependency(self, cand: CandidateEdge, confidence: float, band: str,
                        t_start: float = 0.0, t_end: float = math.inf,
                        weight_share: float = 1.0):
        """weight_share ~ w_ij in Algorithm 1 line 15 (this parent's share
        of the child's incoming dependency mass)."""
        self.G.add_edge(cand.src, cand.dst,
                         layer_pair=cand.layer_pair,
                         confidence=confidence,
                         band=band,
                         t_start=t_start, t_end=t_end,
                         w_ij=weight_share)

    def active_subgraph(self, t: float) -> nx.MultiDiGraph:
        """Edges whose validity window covers time t (contract terminations
        / provider migration silently exit the active graph, per 3.6)."""
        H = nx.MultiDiGraph()
        H.add_nodes_from(self.G.nodes(data=True))
        for u, v, data in self.G.edges(data=True):
            if data["t_start"] <= t <= data["t_end"]:
                H.add_edge(u, v, **data)
        return H

    def band_counts(self) -> Dict[str, int]:
        counts = {"observed": 0, "inferred": 0, "synthetic": 0}
        for _, _, d in self.G.edges(data=True):
            counts[d["band"]] += 1
        return counts


# ======================================================================
# Section 3.7 — Dependency Concentration Encoding (Eqs. 5-9)
# ======================================================================

def k_hop_dependency_sets(G: nx.MultiDiGraph, institutions: List[str],
                           k: int) -> Dict[str, Set[str]]:
    """For each institution, its k-hop reachable dependency set (Eq. 6-7),
    via a standard bounded BFS (established technique, per the paper)."""
    Gr = G.reverse(copy=False)  # we want "what does institution depend on"
    result = {}
    for inst in institutions:
        # institutions "depend on" nodes they point to in the dependency
        # graph; BFS forward from inst up to depth k
        visited = {inst}
        frontier = {inst}
        for _ in range(k):
            nxt = set()
            for n in frontier:
                nxt |= set(G.successors(n))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        result[inst] = visited - {inst}
    return result


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_concentration(mftdg: MFTDG, institutions: List[str],
                           t: float = 0.0,
                           cfg: Config = Config) -> Dict[str, float]:
    """
    Eqs. (5)-(9): for every shared node p, combine
      Conc_p    - raw fan-in (how many institutions reach p within k hops)
      Overlap_p - mean pairwise Jaccard overlap of institutions' k-hop
                  dependency sets restricted through p (the "independent
                  at the application layer, convergent downstream" signal)
      Crit_p    - business-criticality attribute (from node metadata)
      Subst_p   - substitutability: fraction of institutions at p with an
                  immediately available backup (co-located with 3.9's
                  recovery graph); 1 - Subst_p is what enters Eq. (9)
    into a single concentration score DC_p.
    """
    Gt = mftdg.active_subgraph(t)
    dep_sets = k_hop_dependency_sets(Gt, institutions, cfg.K_HOP)

    shared_nodes = [n for n, d in Gt.nodes(data=True)
                    if d.get("layer") not in ("financial_institution",)]

    dc = {}
    for p in shared_nodes:
        reaching = [i for i in institutions if p in dep_sets[i]]
        conc_p = len(reaching) / max(len(institutions), 1)

        if len(reaching) >= 2:
            restricted = {i: {n for n in dep_sets[i] if p in dep_sets[i]}
                          for i in reaching}
            pairs = list(itertools.combinations(reaching, 2))
            overlap_p = float(np.mean([jaccard(restricted[a], restricted[b])
                                        for a, b in pairs])) if pairs else 0.0
        else:
            overlap_p = 0.0

        crit_p = Gt.nodes[p].get("criticality", 0.5)
        subst_p = Gt.nodes[p].get("substitutability", 0.5)  # in [0,1]

        dc[p] = (cfg.LAMBDA["l1"] * conc_p +
                 cfg.LAMBDA["l2"] * overlap_p +
                 cfg.LAMBDA["l3"] * crit_p +
                 cfg.LAMBDA["l4"] * (1 - subst_p))
    return dc  # node -> DC_p, Eq. (9)


# ======================================================================
# Section 3.8 — CI-FCPN: causal multi-stage cascade propagation (Eq. 10-14)
# ======================================================================

@dataclass
class CascadeResult:
    health: Dict[str, List[float]]           # h_j(t) trajectories
    financial_impact: Dict[str, List[float]]  # Y_q(t) trajectories
    timeline: List[float]


class CIFCPN:
    """
    Section 3.8. A disruption is a structural intervention do(X_i = 0) on
    a chosen node (Pearl's SCM / do-calculus, adopted not reinvented):
    the node's own recursion is frozen exogenously at h_i = 0 for the rest
    of the window, and every downstream node updates via the health-state
    recursion in Eq. (11)-(12), scaled by kappa(l) and the substitutability
    discount rho_ij read from ADCRAE's concentration stage (Eq. 8/9) --
    this is the coupling between the concentration stage and the cascade
    stage that the paper calls out explicitly.
    """

    def __init__(self, mftdg: MFTDG, dc_scores: Dict[str, float], cfg: Config = Config):
        self.mftdg = mftdg
        self.dc = dc_scores
        self.cfg = cfg

    def _kappa(self, layer_pair: Tuple[str, str]) -> float:
        return self.cfg.KAPPA.get(layer_pair, self.cfg.KAPPA_DEFAULT)

    def _rho(self, node: str) -> float:
        """Substitutability discount: rho -> 0 if node has an immediately
        available backup (low DC_p / high Subst_p), rho -> 1 if it's a
        single point of failure (high DC_p)."""
        return float(np.clip(self.dc.get(node, 0.0), 0.0, 1.0))

    def propagate(self, intervened: List[str], t_end: int,
                   t0: float = 0.0,
                   processes: Optional[Dict[str, Dict[str, float]]] = None
                   ) -> CascadeResult:
        """
        Implements Algorithm 1, lines 10-18.
        `processes`: {process_q: {"footprint": {node: alpha_iq}, "B_q":.., "F_q":..}}
        """
        Gt = self.mftdg.active_subgraph(t0)
        nodes = list(Gt.nodes())
        order = list(nx.topological_sort(Gt)) if nx.is_directed_acyclic_graph(Gt) \
            else nodes  # graphs with cycles: fall back to given node order

        h = {n: 1.0 for n in nodes}          # health state, 1 = fully healthy
        history = {n: [1.0] for n in nodes}
        intervened_set = set(intervened)

        for n in intervened_set:
            h[n] = 0.0
            history[n] = [0.0]

        timeline = [t0]
        for t in range(1, t_end + 1):
            new_h = dict(h)
            for j in order:
                if j in intervened_set:
                    new_h[j] = 0.0  # frozen exogenously (do-operator)
                    continue
                parents = list(Gt.predecessors(j))
                if not parents:
                    new_h[j] = h[j]
                    continue
                prod_term = 1.0
                for i in parents:
                    for _key, data in Gt.get_edge_data(i, j).items():
                        c_ij = data["confidence"]
                        kappa = self._kappa(data["layer_pair"])
                        rho_ij = self._rho(i)  # discount read from DC_p (Eq.8/9)
                        P_ij = c_ij * kappa * rho_ij           # Eq. (10)
                        w_ij = data.get("w_ij", 1.0)
                        term = (1 - P_ij * (1 - h[i])) ** w_ij
                        prod_term *= max(term, 0.0)
                delta_j = 1 - prod_term                         # Eq. (11)
                new_h[j] = h[j] * (1 - delta_j)                 # Eq. (12)
            h = new_h
            for n in nodes:
                history[n].append(h[n])
            timeline.append(t0 + t)

        financial_impact = {}
        if processes:
            for q, spec in processes.items():
                footprint = spec["footprint"]
                B_q, F_q = spec.get("B_q", 1.0), spec.get("F_q", 1.0)
                series = []
                for step in range(len(timeline)):
                    # Eq. (13): geometric-mean-style impairment; a single
                    # fully failed high-reliance dependency dominates.
                    log_prod = 0.0
                    for i, alpha_iq in footprint.items():
                        h_i = history.get(i, [1.0] * len(timeline))[step]
                        h_i = max(h_i, 1e-9)
                        log_prod += alpha_iq * math.log(h_i)
                    phi_q = 1 - math.exp(log_prod)
                    y_q = B_q * phi_q * F_q                     # Eq. (14)
                    series.append(y_q)
                financial_impact[q] = series

        return CascadeResult(health=history, financial_impact=financial_impact,
                              timeline=timeline)


# ======================================================================
# Section 3.9 — ADCRAE: critical combinations & capacitated recovery
# ======================================================================

class ADCRAE:
    """
    Two responsibilities:
      (a) Eq. (15)-(16): find node combinations whose joint failure is
          disproportionately damaging vs. their individual effects
          (a super-additivity gap), approximated by the classical greedy
          submodular-maximization heuristic (Nemhauser-Wolsey-Fisher,
          1978), seeded by the DC_p ranking from 3.7. No approximation
          guarantee is claimed since Eq. (11) is not shown submodular.
      (b) Eq. (17)-(18): rank recovery pathways as a capacitated
          assignment problem. Uncapacitated -> Hungarian algorithm
          (scipy). Capacitated (shared backups, Eq. 18's constraint) ->
          a capacity-respecting greedy extension of the same assignment,
          which the paper notes would be replaced by a MILP solver once
          capacity limits genuinely bind on real deployments.
    """

    def __init__(self, cifcpn: CIFCPN, dc_scores: Dict[str, float], cfg: Config = Config):
        self.cifcpn = cifcpn
        self.dc = dc_scores
        self.cfg = cfg

    # ---- (a) critical combination search -----------------------------
    def _cascade_impact(self, node_set: List[str], t_end: int,
                         processes: Dict[str, Dict[str, float]]) -> float:
        result = self.cifcpn.propagate(list(node_set), t_end, processes=processes)
        if not result.financial_impact:
            # fall back to total health loss across nodes if no processes given
            return sum(1 - v[-1] for v in result.health.values())
        return sum(series[-1] for series in result.financial_impact.values())

    def find_critical_combinations(self, candidate_nodes: List[str], t_end: int,
                                    processes: Dict[str, Dict[str, float]],
                                    k: int = None) -> List[Tuple[str, ...]]:
        k = k or self.cfg.MAX_COMBO_SIZE
        ranked = sorted(candidate_nodes, key=lambda n: -self.dc.get(n, 0.0))
        singles = {n: self._cascade_impact([n], t_end, processes) for n in ranked}

        best_combo: Tuple[str, ...] = ()
        best_gap = -math.inf
        current: List[str] = []
        results = []
        for _ in range(k):
            best_next, best_next_gap = None, -math.inf
            for n in ranked:
                if n in current:
                    continue
                trial = current + [n]
                impact_joint = self._cascade_impact(trial, t_end, processes)
                impact_sum_singles = sum(singles[m] for m in trial)
                gap = impact_joint - impact_sum_singles          # Eq. (16)
                if gap > best_next_gap:
                    best_next_gap, best_next = gap, n
            if best_next is None:
                break
            current.append(best_next)
            results.append((tuple(current), best_next_gap))
            if best_next_gap > best_gap:
                best_gap, best_combo = best_next_gap, tuple(current)
        return results  # list of (combo, super-additivity gap), growing by size

    # ---- (b) capacitated recovery assignment --------------------------
    def recovery_assignment(self, failed_nodes: List[str],
                             backups: Dict[str, List[str]],
                             ttr: Dict[Tuple[str, str], float],
                             backup_capacity: Optional[Dict[str, int]] = None
                             ) -> Dict[str, Optional[str]]:
        """
        Eq. (17)-(18). `backups[node]` = eligible backup providers for that
        node; `ttr[(node, backup)]` = expected time-to-recovery. Without
        capacity limits this is a plain assignment problem solved exactly
        by the Hungarian algorithm. With shared, limited backup capacity
        (several institutions racing to the same backup) we extend it with
        a capacity-respecting greedy pass, flagged as a heuristic exactly
        as the paper does for the non-guaranteed cases.
        """
        all_backups = sorted({b for opts in backups.values() for b in opts})
        if not all_backups:
            return {n: None for n in failed_nodes}

        if backup_capacity is None:
            # uncapacitated: exact Hungarian algorithm
            cost = np.full((len(failed_nodes), len(all_backups)), fill_value=1e6)
            for i, n in enumerate(failed_nodes):
                for b in backups.get(n, []):
                    j = all_backups.index(b)
                    cost[i, j] = ttr.get((n, b), 1e6)
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {}
            for i, j in zip(row_ind, col_ind):
                assignment[failed_nodes[i]] = (all_backups[j]
                                                if cost[i, j] < 1e6 else None)
            for n in failed_nodes:
                assignment.setdefault(n, None)
            return assignment

        # capacitated: greedy by ascending TTR, respecting remaining capacity
        remaining = dict(backup_capacity)
        assignment = {n: None for n in failed_nodes}
        candidates = [(ttr.get((n, b), 1e6), n, b)
                      for n in failed_nodes for b in backups.get(n, [])]
        candidates.sort()
        assigned = set()
        for cost_val, n, b in candidates:
            if n in assigned:
                continue
            if remaining.get(b, 0) > 0:
                assignment[n] = b
                remaining[b] -= 1
                assigned.add(n)
        return assignment

    @staticmethod
    def resilience_score(assignment: Dict[str, Optional[str]],
                          ttr: Dict[Tuple[str, str], float],
                          business_weight: Dict[str, float],
                          ttr_max: float) -> float:
        """Eq. (17): R_t = 1 - sum(B_j * TTR_j) / sum(B_j * TTR_max)."""
        num, den = 0.0, 0.0
        for n, b in assignment.items():
            B_j = business_weight.get(n, 1.0)
            ttr_j = ttr.get((n, b), ttr_max) if b else ttr_max
            num += B_j * ttr_j
            den += B_j * ttr_max
        return 1 - (num / den if den > 0 else 0.0)


# ======================================================================
# Section 3.10 — Severity output formulation (Eqs. 19-20)
# ======================================================================

def cascade_reach(history: Dict[str, List[float]], step: int,
                   h_min: float = Config.H_MIN) -> float:
    """Eq. (19): fraction of nodes below the failure threshold h_min."""
    vals = [series[step] for series in history.values()]
    failed = sum(1 for v in vals if v < h_min)
    return failed / max(len(vals), 1)


def ficsi(d_cascade: float, resilience: float, normalized_impact: float,
          cfg: Config = Config) -> float:
    """Eq. (20): FICSI_t = g1*D_cascade + g2*(1-R_t) + g3*norm(Y_t)."""
    return (cfg.GAMMA["g1"] * d_cascade +
            cfg.GAMMA["g2"] * (1 - resilience) +
            cfg.GAMMA["g3"] * normalized_impact)


# ======================================================================
# Synthetic ecosystem generator (stands in for FAICID, Sections 3.2-3.3)
# ======================================================================

def generate_synthetic_ecosystem(n_institutions=8, n_ai=3, n_cloud=2, n_data=3,
                                  n_identity=2, n_payment=2, seed=Config.SEED
                                  ) -> Tuple[MFTDG, List[str], Dict[str, Dict]]:
    rng = random.Random(seed)
    mftdg = MFTDG()

    institutions = [f"Bank_{i}" for i in range(n_institutions)]
    ai_services = [f"AIModel_{i}" for i in range(n_ai)]
    clouds = [f"Cloud_{i}" for i in range(n_cloud)]
    data_providers = [f"DataProv_{i}" for i in range(n_data)]
    identities = [f"IdP_{i}" for i in range(n_identity)]
    payments = [f"PaySys_{i}" for i in range(n_payment)]
    applications = [f"App_{i}" for i in range(n_institutions)]
    biz_processes = [f"Process_{i}" for i in range(n_institutions)]
    fin_services = [f"FinService_{i}" for i in range(n_institutions)]

    for i in institutions:
        mftdg.add_entity(i, "financial_institution")
    for a in ai_services:
        mftdg.add_entity(a, "ai_service", criticality=rng.uniform(0.4, 0.9),
                          substitutability=rng.uniform(0.2, 0.7))
    for c in clouds:
        mftdg.add_entity(c, "cloud", criticality=rng.uniform(0.6, 0.95),
                          substitutability=rng.uniform(0.1, 0.4))
    for d in data_providers:
        mftdg.add_entity(d, "data_provider", criticality=rng.uniform(0.3, 0.7),
                          substitutability=rng.uniform(0.3, 0.8))
    for idp in identities:
        mftdg.add_entity(idp, "identity", criticality=rng.uniform(0.5, 0.9),
                          substitutability=rng.uniform(0.2, 0.5))
    for p in payments:
        mftdg.add_entity(p, "payment", criticality=rng.uniform(0.6, 0.95),
                          substitutability=rng.uniform(0.2, 0.5))
    for app, inst in zip(applications, institutions):
        mftdg.add_entity(app, "application", criticality=0.7, substitutability=0.5)
    for bp, inst in zip(biz_processes, institutions):
        mftdg.add_entity(bp, "business_process", criticality=0.8, substitutability=0.4)
    for fs, inst in zip(fin_services, institutions):
        mftdg.add_entity(fs, "financial_service", criticality=0.9, substitutability=0.3)

    estimator = ConfidenceEstimator()
    # small synthetic "manually verified" set to fit Eq. (2) weights (Tier 2)
    verified_X = np.array([[1.0, 0.9, 0.8], [0.9, 0.8, 0.6], [0.3, 0.5, 0.2],
                            [0.2, 0.3, 0.1], [1.0, 1.0, 0.9], [0.4, 0.4, 0.3],
                            [0.95, 0.95, 0.85], [0.85, 0.9, 0.7],
                            [0.15, 0.2, 0.15], [0.25, 0.35, 0.2]])
    verified_y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    estimator.fit(verified_X, verified_y)

    def link(src, dst, layer_pair, primary=False, synthetic=False, agg=None):
        phi_str = rng.uniform(0.85, 1.0) if primary else rng.uniform(0.2, 0.6)
        phi_rel = rng.uniform(0.85, 1.0) if primary else rng.uniform(0.3, 0.7)
        phi_cor = rng.uniform(0.6, 1.0) if primary else rng.uniform(0.2, 0.6)
        cand = CandidateEdge(src, dst, layer_pair, phi_str, phi_rel, phi_cor,
                              is_primary_source=primary,
                              synthetic_target_stat=agg)
        conf = estimator.score(cand)
        band = ConfidenceEstimator.band(conf, primary, synthetic)
        w_ij = rng.uniform(0.5, 1.0)
        mftdg.add_dependency(cand, conf, band, weight_share=w_ij)
        return conf, band

    # institutions -> applications -> business process -> financial service
    for inst, app, bp, fs in zip(institutions, applications, biz_processes, fin_services):
        link(inst, app, ("financial_institution", "application"), primary=True)
        link(app, bp, ("application", "business_process"), primary=True)
        link(bp, fs, ("business_process", "financial_service"), primary=True)
        # each application depends on a subset of shared providers
        for ai in rng.sample(ai_services, k=min(2, len(ai_services))):
            link(ai, app, ("ai_service", "application"),
                 primary=rng.random() > 0.5)
        cloud_choice = rng.choice(clouds)
        link(cloud_choice, app, ("cloud", "application"), primary=True)
        for ai in ai_services:
            link(cloud_choice, ai, ("cloud", "ai_service"), primary=True)
        dp = rng.choice(data_providers)
        link(dp, rng.choice(ai_services), ("data_provider", "ai_service"),
             primary=False, synthetic=(rng.random() < 0.2), agg=0.3)
        idp = rng.choice(identities)
        link(idp, app, ("identity", "application"), primary=True)
        link(cloud_choice, idp, ("cloud", "identity"), primary=True)
        pay = rng.choice(payments)
        link(pay, app, ("payment", "application"), primary=True)
        link(cloud_choice, pay, ("cloud", "payment"), primary=True)

    process_specs = {}
    for bp, fs in zip(biz_processes, fin_services):
        preds = list(mftdg.G.predecessors(bp))
        n = len(preds) if preds else 1
        footprint = {p: 1.0 / n for p in preds}
        process_specs[fs] = dict(footprint=footprint,
                                  B_q=rng.uniform(0.5, 1.0),
                                  F_q=rng.uniform(1e6, 5e7))

    return mftdg, institutions, process_specs


# ======================================================================
# Section 3.11 — Algorithm 1: end-to-end orchestration
# ======================================================================

def run_fid_cif(mftdg: MFTDG, institutions: List[str],
                 process_specs: Dict[str, Dict],
                 scenario_nodes: List[str], t_end: int = 6,
                 cfg: Config = Config) -> Dict:
    """Runs Algorithm 1 end-to-end for one disruption scenario Omega."""
    dc_scores = compute_concentration(mftdg, institutions, cfg=cfg)            # 3.7
    cifcpn = CIFCPN(mftdg, dc_scores, cfg=cfg)                                  # 3.8
    cascade = cifcpn.propagate(scenario_nodes, t_end, processes=process_specs)

    adcrae = ADCRAE(cifcpn, dc_scores, cfg=cfg)                                 # 3.9
    shared_candidates = [n for n in dc_scores if dc_scores[n] > 0][:12]
    combos = adcrae.find_critical_combinations(shared_candidates, t_end,
                                                process_specs, k=cfg.MAX_COMBO_SIZE)

    # toy recovery scenario: each failed node can fail over to same-layer peers
    failed = scenario_nodes
    backups, ttr = {}, {}
    for n in failed:
        layer = mftdg.G.nodes[n]["layer"]
        peers = [m for m, d in mftdg.G.nodes(data=True)
                 if d.get("layer") == layer and m != n]
        backups[n] = peers
        for b in peers:
            ttr[(n, b)] = random.Random(hash((n, b)) % (2**32)).uniform(1, 12)
    capacity = {b: 1 for opts in backups.values() for b in opts}
    assignment = adcrae.recovery_assignment(failed, backups, ttr, capacity)
    biz_weight = {n: mftdg.G.nodes[n].get("criticality", 0.5) for n in failed}
    resilience = adcrae.resilience_score(assignment, ttr, biz_weight, ttr_max=12.0)

    final_step = len(cascade.timeline) - 1
    d_cascade = cascade_reach(cascade.health, final_step, cfg.H_MIN)            # 3.10
    total_impact = sum(series[-1] for series in cascade.financial_impact.values())
    max_possible = sum(spec["B_q"] * spec["F_q"] for spec in process_specs.values())
    norm_impact = total_impact / max_possible if max_possible > 0 else 0.0
    fics = ficsi(d_cascade, resilience, norm_impact, cfg)                       # Eq.(20)

    dominant_path = sorted(cascade.financial_impact.items(),
                            key=lambda kv: -kv[1][-1])[:3]

    return dict(
        dc_scores=dc_scores,
        cascade=cascade,
        critical_combinations=combos,
        recovery_assignment=assignment,
        resilience=resilience,
        cascade_reach=d_cascade,
        financial_impact_total=total_impact,
        ficsi=fics,
        dominant_paths=dominant_path,
        band_counts=mftdg.band_counts(),
    )


# ======================================================================
# Demo entry point
# ======================================================================

if __name__ == "__main__":
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)

    mftdg, institutions, process_specs = generate_synthetic_ecosystem()

    print("=" * 70)
    print("MFTDG constructed.")
    print("Nodes:", mftdg.G.number_of_nodes(), " Edges:", mftdg.G.number_of_edges())
    print("Evidentiary band counts:", mftdg.band_counts())

    # Scenario: a shared cloud provider ("Cloud_0") is disrupted
    scenario_nodes = ["Cloud_0"]
    print("\nDisruption scenario Omega: do(", scenario_nodes, "= failed )")

    results = run_fid_cif(mftdg, institutions, process_specs, scenario_nodes,
                           t_end=6)

    print("\n--- ADCRAE: top-5 most concentrated shared nodes (DC_p) ---")
    for node, score in sorted(results["dc_scores"].items(),
                               key=lambda kv: -kv[1])[:5]:
        print(f"  {node:15s}  DC_p = {score:.3f}")

    print("\n--- CI-FCPN: financial-service health at end of window ---")
    for fs, series in list(results["cascade"].financial_impact.items())[:5]:
        print(f"  {fs:15s}  Y_q(t_end) = ${series[-1]:,.0f}")

    print("\n--- ADCRAE: greedy critical-combination search (Eq. 15-16) ---")
    for combo, gap in results["critical_combinations"]:
        print(f"  combo={combo}  super-additivity gap={gap:,.2f}")

    print("\n--- ADCRAE: capacitated recovery assignment (Eq. 17-18) ---")
    for n, b in results["recovery_assignment"].items():
        print(f"  {n} -> backup: {b}")
    print(f"  Resilience R_t = {results['resilience']:.3f}")

    print("\n--- Section 3.10: composite severity output ---")
    print(f"  Cascade reach D_cascade(t)   = {results['cascade_reach']:.3f}")
    print(f"  Total financial impact Y(t)  = ${results['financial_impact_total']:,.0f}")
    print(f"  FICSI_t                      = {results['ficsi']:.3f}")
    print("  Dominant propagation targets:",
          [d[0] for d in results["dominant_paths"]])
    print("=" * 70)