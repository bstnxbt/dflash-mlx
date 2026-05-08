"""Bet-Optimal Drafting (BOD) — unified chain and tree optimizer.

The draft model is a gambler. The target model is the house.

BOD optimizes the gambler's bet size to maximize throughput. It supports
two betting modes that share the same mathematical core:

Chain mode (vanilla DFlash / standard speculative decoding):
    The gambler lays down γ tokens in a line. Draft cost scales with γ,
    verify cost is fixed. BOD picks the optimal γ.

Tree mode (DDTree / tree-based speculative decoding):
    The gambler lays down B nodes in a tree. Draft cost is fixed (one
    diffusion pass), verify cost scales with B (tree attention). BOD
    picks the optimal B.

The math is identical — both reduce to:

    T(x) = (E[tokens | x] + 1) / (c_fixed + c_scale · x)

where x is the bet size (γ or B), E[tokens] is a concave increasing
function of x, and the denominator is linear. The optimal x maximizes
this ratio. The costs simply swap roles:

    Chain: c_fixed = c_verify,  c_scale = c_draft_per_token,  x = γ
    Tree:  c_fixed = c_draft,   c_scale = c_verify_per_node,  x = B

Each mode has three optimization tiers, from cheapest to most general:

Chain mode:
    1. Verify-dominated (c_v >> c_d·γ_max): return max γ immediately.
       The house fee dwarfs the dealing cost, so every extra card
       amortizes the fixed overhead. One comparison, zero math. Common
       on hybrid SSM-attention architectures (Qwen3.5).
    2. ρ = 0 (no Markov recovery): closed-form via Lambert W.
       FOC reduces to (u+K)·exp(−u) = 1 with K = 1 + (c_v/c_d)·|ln α|.
       Solution: u = −W₋₁(−e⁻ᴷ) − K, asymptotic u ≈ ln K for K > 30.
       One log, one Lambert W, no GPU. Common case for standard spec decode.
    3. ρ > 0 (Markov recovery): fused Metal kernel sweep.
       The unified Markov formula ea = n·ρ/d + λ·(1−α)·(1−λⁿ)/d² has
       no known closed-form argmax. One thread per candidate γ evaluates
       all candidates in a single GPU dispatch. Only reached with extended
       verification schemes.

Tree mode:
    1. Draft-dominated (c_draft >> c_vpn·B_max): return max B immediately.
       Verify per node is negligible, so always expand the tree. One
       comparison, zero math. Common when tree attention is heavily
       optimized or the diffusion pass is slow.
    2. Enough observations: closed-form via Lambert W on log model.
       E[accepted] ≈ a·ln(B) + b, fit online from observed (B, τ) pairs.
       FOC gives B* = R / W₀(R·e^{k/a}) where R = c_draft/c_vpn.
       One log, one Lambert W, no GPU.
    3. Cold start (insufficient data): fused Metal kernel sweep.
       Evaluates throughput at all candidate budgets using the default
       log-acceptance model. One GPU dispatch.

Usage (chain mode — vanilla DFlash):
    bod = BODController(BODConfig(mode='chain'))
    for cycle in spec_decode_loop:
        gamma = bod.optimal_bet()
        # ... draft gamma tokens, verify ...
        bod.observe(bet=gamma, accepted=n_accepted, ...)

Usage (tree mode — DDTree):
    bod = BODController(BODConfig(mode='tree'))
    for cycle in spec_decode_loop:
        budget = bod.optimal_bet()
        # ... build tree with budget nodes, verify ...
        bod.observe(bet=budget, accepted=n_accepted, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx


# ---------------------------------------------------------------------------
# Metal kernel — fused Markov throughput sweep (chain mode only)
# ---------------------------------------------------------------------------

_BOD_CHAIN_KERNEL_SOURCE = """
    uint idx = thread_position_in_grid.x;

    float alpha         = params[0];
    float rho           = params[1];
    float draft_per_tok = params[2];
    float verify_cost   = params[3];
    float min_gamma     = params[4];

    float gamma = min_gamma + (float)idx;
    float n = gamma - 1.0f;

    float ea;
    if (alpha > 0.999f) {
        ea = n;
    } else if (alpha < 0.001f) {
        ea = 0.0f;
    } else {
        float lam = alpha - rho;
        float d = 1.0f - lam;
        float lam_pow_n = metal::pow(lam, n);
        ea = n * rho / d
           + lam * (1.0f - alpha) * (1.0f - lam_pow_n) / (d * d);
    }

    float cycle_time = draft_per_tok * gamma + verify_cost;
    throughputs[idx] = (ea + 1.0f) / cycle_time;
"""

_bod_chain_kernel = mx.fast.metal_kernel(
    name="bod_chain_markov",
    input_names=["params"],
    output_names=["throughputs"],
    source=_BOD_CHAIN_KERNEL_SOURCE,
)


# ---------------------------------------------------------------------------
# Metal kernel — fused log-acceptance throughput sweep (tree mode)
# ---------------------------------------------------------------------------

_BOD_TREE_KERNEL_SOURCE = """
    uint idx = thread_position_in_grid.x;

    float log_slope  = params[0];
    float log_offset = params[1];
    float c_fixed    = params[2];
    float c_scale    = params[3];
    float min_bet    = params[4];

    float bet = min_bet + (float)idx;

    // E[tokens] + 1 = a * ln(bet) + b + 1
    float ea_plus_1 = log_slope * metal::log(bet) + log_offset + 1.0f;
    if (ea_plus_1 < 1.0f) ea_plus_1 = 1.0f;  // floor: at least the bonus token

    float cycle_time = c_fixed + c_scale * bet;
    throughputs[idx] = ea_plus_1 / cycle_time;
"""

_bod_tree_kernel = mx.fast.metal_kernel(
    name="bod_tree_log",
    input_names=["params"],
    output_names=["throughputs"],
    source=_BOD_TREE_KERNEL_SOURCE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BODObservation:
    """One cycle's worth of data. Fields used depend on mode."""
    bet: int               # γ (chain) or B (tree)
    accepted: int          # tokens accepted this cycle
    cycle_time_ms: float   # total wall time
    draft_time_ms: float   # draft model time
    verify_time_ms: float  # target model time
    confidence: float      # draft model avg max-softmax


@dataclass
class BODConfig:
    """Configuration for BOD controller.

    mode='chain': optimize draft length γ (vanilla DFlash).
    mode='tree':  optimize node budget B (DDTree).
    """
    mode: str = 'chain'  # 'chain' or 'tree'

    # -- Bet range --
    min_bet: int = 2
    max_bet: int = 16     # γ_max (chain) or B_max (tree)

    # -- Default costs (before enough data) --
    default_scale_cost: float = 8.0   # c_d per token (chain) or c_vpn (tree)
    default_fixed_cost: float = 47.0  # c_v (chain) or c_draft (tree)

    # -- Observation windows --
    acceptance_window: int = 8
    cost_window: int = 16
    cost_min_observations: int = 4
    cost_min_unique_bets: int = 2
    max_obs_age: int = 64

    # -- Chain-mode Markov model --
    confidence_weight: float = 0.3
    rho: float = 0.0

    # -- Tree-mode log acceptance model --
    # E[accepted] ≈ log_slope * ln(B) + log_offset
    # Estimated online from observed (B, tau) pairs.
    # Defaults are from MATH-500 / Qwen3-8B empirical fit.
    default_log_slope: float = 0.71
    default_log_offset: float = 5.18

    # -- Analytical fast path --
    dominance_ratio: float = 4.0

    # -- Tree-mode budget candidates --
    # DDTree only tests powers of 2. BOD can sweep any range, but for
    # compatibility with DDTree we default to the same set.
    tree_budget_candidates: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class BODController:
    """Unified bet-size optimizer for chain and tree speculative decoding.

    Chain mode fast paths:
    1. Verify-dominated → max_bet.
    2. ρ = 0 → Lambert W closed form.
    3. ρ > 0 → Metal kernel sweep.

    Tree mode fast paths:
    1. Draft-dominated → max_bet (tree attention is cheap, always expand).
    2. Enough observations → Lambert W on log acceptance model.
    3. Fallback → Metal kernel sweep over budget candidates.
    """

    def __init__(self, cfg: BODConfig | None = None):
        self.cfg = cfg or BODConfig()
        self._observations: list[BODObservation] = []
        self._cycle: int = 0
        self._rho_estimate: float = self.cfg.rho

        # Tree mode: online log-acceptance model
        self._log_slope: float = self.cfg.default_log_slope
        self._log_offset: float = self.cfg.default_log_offset

    @property
    def cycle(self) -> int:
        return self._cycle

    def optimal_bet(self) -> int:
        """Return the optimal bet size for the next cycle."""
        if self.cfg.mode == 'chain':
            return self._optimal_chain()
        return self._optimal_tree()

    def observe(
        self,
        bet: int,
        accepted: int,
        cycle_time_ms: float,
        draft_time_ms: float,
        verify_time_ms: float,
        confidence: float = 0.0,
    ) -> None:
        """Record what happened this cycle."""
        self._observations.append(BODObservation(
            bet=bet, accepted=accepted, cycle_time_ms=cycle_time_ms,
            draft_time_ms=draft_time_ms, verify_time_ms=verify_time_ms,
            confidence=confidence,
        ))
        self._cycle += 1
        if len(self._observations) > self.cfg.max_obs_age:
            self._observations = self._observations[-self.cfg.max_obs_age:]

        if self.cfg.mode == 'tree':
            self._update_log_model()

    # -------------------------------------------------------------------
    # Chain mode (DFlash)
    # -------------------------------------------------------------------

    def _optimal_chain(self) -> int:
        cfg = self.cfg
        alpha = self._estimate_alpha()
        c_scale, c_fixed = self._estimate_costs_chain()

        # Fast path 1: verify-dominated → max γ
        if c_fixed > cfg.dominance_ratio * c_scale * cfg.max_bet:
            return cfg.max_bet

        # Fast path 2: ρ=0 → Lambert W
        if self._rho_estimate < 1e-8:
            return _analytical_gamma(
                alpha, c_scale, c_fixed, cfg.min_bet, cfg.max_bet,
            )

        # General: Metal kernel sweep
        return _chain_sweep_metal(
            alpha, self._rho_estimate, c_scale, c_fixed,
            cfg.min_bet, cfg.max_bet,
        )

    def _estimate_alpha(self) -> float:
        """Confidence-weighted acceptance rate (chain mode)."""
        w = self.cfg.confidence_weight
        obs = self._observations
        if not obs:
            return 0.85
        recent = obs[-self.cfg.acceptance_window:]
        rates = [o.accepted / max(o.bet - 1, 1) for o in recent]
        observed = sum(rates) / len(rates)
        confs = [o.confidence for o in recent if o.confidence > 0]
        conf_alpha = sum(confs) / len(confs) if confs else observed
        blended = (1.0 - w) * observed + w * conf_alpha
        return max(0.01, min(blended, 0.999))

    def _estimate_costs_chain(self) -> tuple[float, float]:
        """(c_scale=draft_per_tok, c_fixed=verify_cost) for chain mode."""
        obs = self._observations
        if len(obs) < self.cfg.cost_min_observations:
            return self.cfg.default_scale_cost, self.cfg.default_fixed_cost
        recent = obs[-self.cfg.cost_window:]
        if len(set(o.bet for o in recent)) < self.cfg.cost_min_unique_bets:
            return self.cfg.default_scale_cost, self.cfg.default_fixed_cost

        c_scale = _ols_slope([float(o.bet) for o in recent],
                             [o.draft_time_ms for o in recent])
        c_scale = max(c_scale, 0.5)

        c_fixed = _median([o.verify_time_ms for o in recent])
        c_fixed = max(c_fixed, 1.0)
        return c_scale, c_fixed

    # -------------------------------------------------------------------
    # Tree mode (DDTree)
    # -------------------------------------------------------------------

    def _optimal_tree(self) -> int:
        cfg = self.cfg
        c_scale, c_fixed = self._estimate_costs_tree()

        # Fast path 1: draft-dominated → max budget
        # (verify per node is negligible, always expand the tree)
        if c_fixed > cfg.dominance_ratio * c_scale * cfg.max_bet:
            return cfg.max_bet

        # Fast path 2: enough data → Lambert W on log model
        if len(self._observations) >= cfg.cost_min_observations:
            return _analytical_budget(
                self._log_slope, self._log_offset,
                c_scale, c_fixed, cfg.min_bet, cfg.max_bet,
            )

        # Fallback: Metal kernel sweep over candidates
        return _tree_sweep_metal(
            self._log_slope, self._log_offset,
            c_scale, c_fixed, cfg.min_bet, cfg.max_bet,
        )

    def _estimate_costs_tree(self) -> tuple[float, float]:
        """(c_scale=verify_per_node, c_fixed=draft_cost) for tree mode.

        Costs swap vs chain: verify scales with B, draft is fixed.
        """
        obs = self._observations
        if len(obs) < self.cfg.cost_min_observations:
            return self.cfg.default_scale_cost, self.cfg.default_fixed_cost
        recent = obs[-self.cfg.cost_window:]
        if len(set(o.bet for o in recent)) < self.cfg.cost_min_unique_bets:
            return self.cfg.default_scale_cost, self.cfg.default_fixed_cost

        # Verify cost scales with B → OLS on (B, verify_time)
        c_scale = _ols_slope([float(o.bet) for o in recent],
                             [o.verify_time_ms for o in recent])
        c_scale = max(c_scale, 0.001)

        # Draft cost is fixed → median of draft times
        c_fixed = _median([o.draft_time_ms for o in recent])
        c_fixed = max(c_fixed, 1.0)
        return c_scale, c_fixed

    def _update_log_model(self) -> None:
        """Fit S(B) ≈ a·ln(B) + b from observed (bet, accepted) pairs."""
        obs = self._observations
        if len(obs) < self.cfg.cost_min_observations:
            return
        recent = obs[-self.cfg.cost_window:]
        if len(set(o.bet for o in recent)) < self.cfg.cost_min_unique_bets:
            return

        # OLS on (ln(B), tau) where tau = accepted + 1
        log_bets = [math.log(o.bet) for o in recent]
        taus = [float(o.accepted + 1) for o in recent]
        n = len(log_bets)
        sx = sum(log_bets); sy = sum(taus)
        sxx = sum(x * x for x in log_bets)
        sxy = sum(x * y for x, y in zip(log_bets, taus))
        denom = n * sxx - sx * sx
        if abs(denom) < 0.001:
            return
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        # Sanity: slope should be positive (more nodes → more acceptance)
        if slope > 0.01:
            self._log_slope = slope
            self._log_offset = intercept


# ---------------------------------------------------------------------------
# Analytical solvers (pure Python — no GPU)
# ---------------------------------------------------------------------------

def _analytical_gamma(
    alpha: float, c_scale: float, c_fixed: float,
    min_bet: int = 2, max_bet: int = 16,
) -> int:
    """Closed-form optimal γ for chain mode (ρ=0) via Lambert W.

    FOC: (u + K)·exp(−u) = 1, K = 1 + (c_fixed/c_scale)·|ln α|.
    Solution: u = −W₋₁(−e⁻ᴷ) − K, γ = u / |ln α|.
    Asymptotic fallback u ≈ ln(K) when K > 30.
    """
    if alpha <= 0.001: return min_bet
    if alpha >= 0.999: return max_bet
    if c_scale < 1e-10: return max_bet

    L = -math.log(alpha)
    K = 1.0 + (c_fixed / c_scale) * L

    if K > 30:
        u = math.log(K)
    else:
        try:
            from scipy.special import lambertw
            import numpy as np
            w_val = float(np.real(lambertw(-math.exp(-K), k=-1)))
            u = -w_val - K
        except ImportError:
            u = math.log(K)

    gamma = u / L
    if not math.isfinite(gamma): return max_bet
    return max(min_bet, min(int(round(gamma)), max_bet))


def _analytical_budget(
    log_slope: float, log_offset: float,
    c_scale: float, c_fixed: float,
    min_bet: int = 16, max_bet: int = 1024,
) -> int:
    """Closed-form optimal B for tree mode via Lambert W.

    Throughput T(B) = (a·ln(B) + b + 1) / (c_fixed + c_scale·B).
    FOC: B* = R / W₀(R · e^{k/a}), R = c_fixed/c_scale, k = b + 1 - a.
    """
    if c_scale < 1e-10: return max_bet
    if log_slope < 0.01: return min_bet

    a = log_slope
    k = log_offset + 1.0 - a
    R = c_fixed / c_scale

    z = R * math.exp(k / a)
    if not math.isfinite(z) or z <= 0:
        return max_bet

    try:
        from scipy.special import lambertw
        import numpy as np
        w_val = float(np.real(lambertw(z, k=0)))
        if w_val <= 0: return max_bet
        B_star = R / w_val
    except ImportError:
        # Asymptotic: W₀(z) ≈ ln(z) - ln(ln(z)) for large z
        ln_z = math.log(z)
        w_approx = ln_z - math.log(ln_z) if ln_z > 1 else 1.0
        B_star = R / w_approx

    if not math.isfinite(B_star): return max_bet
    return max(min_bet, min(int(round(B_star)), max_bet))


# ---------------------------------------------------------------------------
# Metal kernel sweeps
# ---------------------------------------------------------------------------

def _chain_sweep_metal(
    alpha: float, rho: float,
    c_scale: float, c_fixed: float,
    min_bet: int, max_bet: int,
) -> int:
    """Sweep γ candidates via fused Metal kernel (chain mode, ρ > 0)."""
    num = max_bet - min_bet + 1
    params = mx.array(
        [alpha, rho, c_scale, c_fixed, float(min_bet)],
        dtype=mx.float32,
    )
    outputs = _bod_chain_kernel(
        inputs=[params],
        template=[("T", mx.float32)],
        grid=(num, 1, 1), threadgroup=(num, 1, 1),
        output_shapes=[(num,)], output_dtypes=[mx.float32],
    )
    return min_bet + int(mx.argmax(outputs[0]))


def _tree_sweep_metal(
    log_slope: float, log_offset: float,
    c_scale: float, c_fixed: float,
    min_bet: int, max_bet: int,
) -> int:
    """Sweep B candidates via fused Metal kernel (tree mode)."""
    num = max_bet - min_bet + 1
    params = mx.array(
        [log_slope, log_offset, c_fixed, c_scale, float(min_bet)],
        dtype=mx.float32,
    )
    outputs = _bod_tree_kernel(
        inputs=[params],
        template=[("T", mx.float32)],
        grid=(num, 1, 1), threadgroup=(min(num, 256), 1, 1),
        output_shapes=[(num,)], output_dtypes=[mx.float32],
    )
    return min_bet + int(mx.argmax(outputs[0]))


# ---------------------------------------------------------------------------
# Pure Python helpers
# ---------------------------------------------------------------------------

def _ols_slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope on ≤16 observations. Pure Python."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 0.001: return 8.0
    return (n * sxy - sx * sy) / denom


def _median(xs: list[float]) -> float:
    """Median of a small list. Pure Python."""
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Pure-MLX fallbacks
# ---------------------------------------------------------------------------

def _expected_accepted_markov(
    alpha: mx.array, rho: float, ns: mx.array,
) -> mx.array:
    """Markov expected acceptance — pure MLX reference."""
    lam = alpha - rho
    d = 1.0 - lam + 1e-10
    lam_pow_n = lam ** ns
    return ns * rho / d + lam * (1.0 - alpha) * (1.0 - lam_pow_n) / (d * d)


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------

def bod_optimal_bet(
    mode: str = 'chain',
    alpha: float = 0.85,
    c_scale: float = 8.0,
    c_fixed: float = 47.0,
    rho: float = 0.0,
    log_slope: float = 0.71,
    log_offset: float = 5.18,
    min_bet: int = 2,
    max_bet: int = 16,
) -> int:
    """Stateless convenience — pick optimal bet given known parameters.

    Chain mode:
    >>> bod_optimal_bet('chain', alpha=0.85, c_scale=8.0, c_fixed=47.0)
    7
    >>> bod_optimal_bet('chain', alpha=0.95, c_scale=8.0, c_fixed=47.0)
    13

    Tree mode:
    >>> bod_optimal_bet('tree', c_scale=0.05, c_fixed=100.0,
    ...                 log_slope=0.71, log_offset=5.18,
    ...                 min_bet=16, max_bet=1024)
    512
    """
    if mode == 'chain':
        if rho < 1e-8:
            return _analytical_gamma(alpha, c_scale, c_fixed, min_bet, max_bet)
        return _chain_sweep_metal(
            alpha, rho, c_scale, c_fixed, min_bet, max_bet,
        )
    else:
        return _analytical_budget(
            log_slope, log_offset, c_scale, c_fixed, min_bet, max_bet,
        )
