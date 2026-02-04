#!/usr/bin/env python3
"""
Markov Chain Analysis Toolkit — Structural Inequality Simulator
================================================================

A reusable tool for designing and validating Markov chain parameters
for the structural racism persistence simulation.

Usage:
    # Quick validation of a specific config:
    python markov_toolkit.py --edge 0.10 --near-edge 0.20 --middle 0.25

    # Full parameter sweep:
    python markov_toolkit.py --sweep

    # Full scenario simulation with specific barrier duration:
    python markov_toolkit.py --edge 0.10 --near-edge 0.20 --middle 0.25 --barrier-gens 30

    # Generate visualization charts:
    python markov_toolkit.py --edge 0.10 --near-edge 0.20 --middle 0.25 --barrier-gens 30 --plot

    # Custom population and convergence threshold:
    python markov_toolkit.py --edge 0.10 --near-edge 0.20 --middle 0.25 --n-black 50 --n-white 50 --threshold 0.15

    # Export results to JSON:
    python markov_toolkit.py --edge 0.10 --near-edge 0.20 --middle 0.25 --barrier-gens 30 --export results.json

Theory:
    For a 7-state chain with adjacent-only transitions and a UNIFORM
    stationary distribution (1/7 per state), detailed balance requires:
        P(up from state i) = P(down from state i+1)

    We parameterize the chain with three "mobility" values:
        - edge:      u_1 = d_2  and  u_6 = d_7   (extremes)
        - near_edge: u_2 = d_3  and  u_5 = d_6   (near-extremes)
        - middle:    u_3 = d_4  and  u_4 = d_5   (center)

    This guarantees uniform stationary distribution by construction.
    The spectral gap (1 - second_eigenvalue) controls mixing speed.
"""

import argparse
import json
import sys
import numpy as np
from numpy.linalg import eig


# ============================================================
# CORE: Transition Matrix Construction
# ============================================================

TIER_NAMES = [
    'Destitute', 'Poor', 'Working', 'Middle',
    'Upper Mid', 'Wealthy', 'Ultra Wealthy'
]

def build_transition_matrix(edge: float, near_edge: float, middle: float) -> np.ndarray:
    """
    Build a 7x7 transition matrix satisfying detailed balance
    for uniform stationary distribution.

    Parameters:
        edge:      Mobility at extremes (tiers 1↔2 and 6↔7)
        near_edge: Mobility near extremes (tiers 2↔3 and 5↔6)
        middle:    Mobility in the center (tiers 3↔4 and 4↔5)

    Returns:
        7x7 numpy array where T[i,j] = P(move from tier i to tier j)
    """
    # Up probabilities: u[i] = P(move up from tier i)
    # By detailed balance for uniform dist: u[i] = d[i+1]
    u = [edge, near_edge, middle, middle, near_edge, edge, 0.0]
    d = [0.0] + u[:-1]

    T = np.zeros((7, 7))
    for i in range(7):
        stay = 1.0 - u[i] - d[i]
        if stay < 0:
            raise ValueError(
                f"Invalid params: tier {i+1} has negative stay probability "
                f"({stay:.4f}). Reduce mobility values."
            )
        T[i, i] = stay
        if i < 6:
            T[i, i + 1] = u[i]
        if i > 0:
            T[i, i - 1] = d[i]

    return T


# ============================================================
# ANALYSIS: Stationary Distribution & Spectral Properties
# ============================================================

def analyze_chain(T: np.ndarray) -> dict:
    """
    Compute stationary distribution, eigenvalues, spectral gap,
    and mixing time estimate for a transition matrix.

    Returns dict with all analysis results.
    """
    eigenvalues, eigenvectors = eig(T.T)

    # Find eigenvector for eigenvalue ≈ 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()

    # Spectral gap
    evals_real = np.sort(np.abs(np.real(eigenvalues)))[::-1]
    second_eval = evals_real[1]
    spectral_gap = 1.0 - second_eval

    return {
        'stationary': stationary,
        'eigenvalues': np.sort(np.real(eigenvalues))[::-1],
        'second_eigenvalue': second_eval,
        'spectral_gap': spectral_gap,
        'mixing_time': 1.0 / spectral_gap if spectral_gap > 0 else float('inf'),
        'is_uniform': np.max(np.abs(stationary - 1.0 / 7)) < 1e-6,
        'max_deviation_from_uniform': np.max(np.abs(stationary - 1.0 / 7)),
    }


# ============================================================
# SIMULATION: Full Scenario
# ============================================================

def simulate_scenario(
    T: np.ndarray,
    n_black: int = 50,
    n_white: int = 50,
    barrier_gens: int = 30,
    barrier_cap: int = 2,  # 0-indexed tier (tier 3 = index 2)
    max_gen: int = 300,
    n_trials: int = 500,
    convergence_threshold: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Run the full discrimination → liberation → convergence simulation.

    Parameters:
        T: Transition matrix
        n_black, n_white: Population sizes
        barrier_gens: How many generations discrimination lasts
        barrier_cap: Max tier for black dots during discrimination (0-indexed)
        max_gen: Maximum generations to simulate
        n_trials: Number of Monte Carlo trials
        convergence_threshold: Max deviation from 50% representation per tier
        seed: Random seed for reproducibility

    Returns:
        Dict with gap trajectory, convergence stats, distribution snapshots
    """
    rng = np.random.default_rng(seed)
    n_total = n_black + n_white

    all_gaps = []
    convergence_gens = []
    all_black_dists = []
    all_white_dists = []

    for trial in range(n_trials):
        black = np.zeros(n_black, dtype=int)
        white = rng.integers(0, 7, n_white)

        gaps = []
        b_dists = []
        w_dists = []
        converged_gen = None

        for gen in range(max_gen):
            # Record stats
            gaps.append(float(white.mean() - black.mean()))
            b_dists.append(np.bincount(black, minlength=7).tolist())
            w_dists.append(np.bincount(white, minlength=7).tolist())

            # Check convergence post-liberation
            if gen > barrier_gens and converged_gen is None:
                all_converged = True
                for tier in range(7):
                    b_count = np.sum(black == tier)
                    total = b_count + np.sum(white == tier)
                    if total > 3:
                        b_frac = b_count / total
                        if abs(b_frac - 0.5) > convergence_threshold:
                            all_converged = False
                            break
                if all_converged:
                    converged_gen = gen - barrier_gens

            # Step
            is_barrier = gen < barrier_gens

            # Move black dots
            new_black = black.copy()
            for i in range(n_black):
                r = rng.random()
                cum = 0.0
                for j in range(7):
                    if is_barrier and j > barrier_cap:
                        break
                    cum += T[black[i], j]
                    if r < cum:
                        new_black[i] = j
                        break
                if is_barrier and new_black[i] > barrier_cap:
                    new_black[i] = barrier_cap
            black = new_black

            # Move white dots
            new_white = white.copy()
            for i in range(n_white):
                r = rng.random()
                cum = 0.0
                for j in range(7):
                    cum += T[white[i], j]
                    if r < cum:
                        new_white[i] = j
                        break
            white = new_white

        all_gaps.append(gaps)
        all_black_dists.append(b_dists)
        all_white_dists.append(w_dists)
        convergence_gens.append(
            converged_gen if converged_gen is not None else max_gen - barrier_gens
        )

    avg_gaps = np.mean(all_gaps, axis=0)
    avg_b_dists = np.mean(all_black_dists, axis=0)
    avg_w_dists = np.mean(all_white_dists, axis=0)

    # Find threshold crossings
    thresholds = {}
    for threshold in [1.0, 0.5, 0.25, 0.10]:
        for g in range(barrier_gens + 1, len(avg_gaps)):
            if avg_gaps[g] < threshold:
                thresholds[f'gap_below_{threshold}'] = {
                    'generation': int(g),
                    'gens_after_liberation': int(g - barrier_gens),
                }
                break

    return {
        'avg_gap_trajectory': avg_gaps.tolist(),
        'avg_black_distributions': avg_b_dists.tolist(),
        'avg_white_distributions': avg_w_dists.tolist(),
        'convergence': {
            'mean': float(np.mean(convergence_gens)),
            'median': float(np.median(convergence_gens)),
            'p90': float(np.percentile(convergence_gens, 90)),
            'p95': float(np.percentile(convergence_gens, 95)),
            'max': int(np.max(convergence_gens)),
        },
        'gap_at_liberation': float(avg_gaps[barrier_gens]),
        'gap_milestones': {
            f'+{d}_gens': float(avg_gaps[min(barrier_gens + d, max_gen - 1)])
            for d in [10, 20, 50, 100, 150, 200]
            if barrier_gens + d < max_gen
        },
        'threshold_crossings': thresholds,
    }


# ============================================================
# PARAMETER SWEEP
# ============================================================

def parameter_sweep(
    configs: list[tuple[str, float, float, float]] = None,
    barrier_gens: int = 30,
    n_trials: int = 200,
) -> list[dict]:
    """
    Sweep over multiple parameter configurations and compare.
    """
    if configs is None:
        configs = [
            ('Very Sticky', 0.05, 0.10, 0.15),
            ('Sticky', 0.10, 0.20, 0.25),
            ('Moderate', 0.15, 0.25, 0.30),
            ('Mobile', 0.20, 0.30, 0.35),
        ]

    results = []
    for name, edge, near_edge, middle in configs:
        T = build_transition_matrix(edge, near_edge, middle)
        analysis = analyze_chain(T)
        sim = simulate_scenario(
            T, barrier_gens=barrier_gens, n_trials=n_trials, max_gen=400
        )
        results.append({
            'name': name,
            'params': {'edge': edge, 'near_edge': near_edge, 'middle': middle},
            'is_uniform': analysis['is_uniform'],
            'spectral_gap': analysis['spectral_gap'],
            'mixing_time': analysis['mixing_time'],
            'convergence_mean': sim['convergence']['mean'],
            'convergence_median': sim['convergence']['median'],
            'gap_at_liberation': sim['gap_at_liberation'],
        })

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def generate_plots(
    T: np.ndarray,
    sim_results: dict,
    barrier_gens: int = 30,
    params: dict = None,
    output_dir: str = '.',
):
    """Generate analysis charts. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        print("Install with: pip install matplotlib")
        return

    avg_gaps = sim_results['avg_gap_trajectory']
    avg_b = np.array(sim_results['avg_black_distributions'])
    avg_w = np.array(sim_results['avg_white_distributions'])
    max_gen = len(avg_gaps)
    param_label = (
        f"{params['edge']}/{params['near_edge']}/{params['middle']}"
        if params else "custom"
    )

    # ---- Figure 1: Gap trajectory with annotations ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]}
    )

    gens = np.arange(max_gen)

    ax1.fill_between(range(barrier_gens), -0.5, 3.5, alpha=0.08, color='red',
                     label='Discrimination era')
    ax1.axvline(x=barrier_gens, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax1.plot(gens, avg_gaps, color='#2c3e50', linewidth=2.5)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax1.annotate(
        'Barriers removed!',
        xy=(barrier_gens, avg_gaps[barrier_gens]),
        xytext=(barrier_gens + 15, 2.8),
        fontsize=11, fontweight='bold', color='#e74c3c',
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
    )

    # Annotate gap milestones
    for offset, color in [(50, '#e67e22'), (100, '#2ecc71')]:
        g = barrier_gens + offset
        if g < max_gen:
            ax1.annotate(
                f'+{offset} gens:\ngap = {avg_gaps[g]:.2f}',
                xy=(g, avg_gaps[g]),
                xytext=(g + 10, avg_gaps[g] + 0.5),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='->', color=color),
            )

    ax1.set_ylabel('Wealth Gap (tiers)', fontsize=12)
    ax1.set_title(
        f'Structural Inequality Persistence — params: {param_label}',
        fontsize=14, fontweight='bold',
    )
    ax1.set_ylim(-0.5, 3.5)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Bottom: average tier per group
    ax2.fill_between(range(barrier_gens), 0, 7, alpha=0.08, color='red')
    ax2.axvline(x=barrier_gens, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)

    black_avg = [avg_b[g] @ np.arange(7) / 50 for g in range(max_gen)]
    white_avg = [avg_w[g] @ np.arange(7) / 50 for g in range(max_gen)]

    ax2.plot(gens, black_avg, color='#2c3e50', linewidth=2, label='Black avg tier')
    ax2.plot(gens, white_avg, color='#95a5a6', linewidth=2, label='White avg tier')
    ax2.axhline(y=3, color='gray', linestyle=':', alpha=0.5, label='Equal (tier 3)')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Avg Tier', fontsize=12)
    ax2.set_ylim(0, 6)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{output_dir}/gap_trajectory.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ---- Figure 2: Distribution snapshots ----
    snapshots = [0, 10, 29, 30, 50, 80, 120, 180]
    snapshots = [s for s in snapshots if s < max_gen]
    n_snaps = len(snapshots)
    cols = min(4, n_snaps)
    rows = (n_snaps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    fig.suptitle(
        f'Tier Distribution Snapshots — params: {param_label}',
        fontsize=14, fontweight='bold',
    )

    for idx, gen in enumerate(snapshots):
        ax = axes[idx] if n_snaps > 1 else axes
        b = avg_b[gen]
        w = avg_w[gen]
        x = np.arange(7)
        width = 0.35

        ax.bar(x - width / 2, b, width, label='Black', color='#2c3e50', alpha=0.85)
        ax.bar(x + width / 2, w, width, label='White', color='#bdc3c7',
               edgecolor='#7f8c8d', alpha=0.85)

        phase = (
            'DISCRIMINATION' if gen < barrier_gens
            else ('LIBERATION!' if gen == barrier_gens
                  else f'Post-lib +{gen - barrier_gens}')
        )
        color = '#e74c3c' if gen < barrier_gens else (
            '#27ae60' if gen == barrier_gens else '#3498db'
        )
        ax.set_title(f'Gen {gen}: {phase}', fontsize=10, color=color, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n[:6] for n in TIER_NAMES], fontsize=7, rotation=45)
        ax.set_ylim(0, 52)
        ax.set_ylabel('Avg # of dots')
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(len(snapshots), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = f'{output_dir}/distribution_snapshots.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# PRINTING UTILITIES
# ============================================================

def print_matrix(T: np.ndarray):
    """Pretty-print the transition matrix."""
    print(f"\n{'Tier':<17} {'Down':>8} {'Stay':>8} {'Up':>8}")
    print('-' * 44)
    for i in range(7):
        down = T[i, i - 1] if i > 0 else 0
        stay = T[i, i]
        up = T[i, i + 1] if i < 6 else 0
        print(f"  {i+1}. {TIER_NAMES[i]:<12} {down:>8.2f} {stay:>8.2f} {up:>8.2f}")


def print_analysis(analysis: dict):
    """Pretty-print chain analysis results."""
    print(f"\n  Stationary distribution uniform: {analysis['is_uniform']}")
    print(f"  Max deviation from 1/7: {analysis['max_deviation_from_uniform']:.2e}")
    print(f"  Spectral gap: {analysis['spectral_gap']:.4f}")
    print(f"  Mixing time estimate: ~{analysis['mixing_time']:.0f} generations")
    print(f"  Eigenvalues: {analysis['eigenvalues']}")


def print_scenario(sim: dict, barrier_gens: int):
    """Pretty-print scenario simulation results."""
    print(f"\n  Gap at liberation (gen {barrier_gens}): {sim['gap_at_liberation']:.2f} tiers")
    print(f"\n  Gap over time:")
    for label, val in sim['gap_milestones'].items():
        print(f"    {label}: {val:.3f}")
    print(f"\n  Convergence (gens post-liberation):")
    for k, v in sim['convergence'].items():
        print(f"    {k}: {v:.1f}" if isinstance(v, float) else f"    {k}: {v}")
    if sim['threshold_crossings']:
        print(f"\n  Gap threshold crossings:")
        for label, info in sim['threshold_crossings'].items():
            print(f"    {label}: gen {info['generation']} ({info['gens_after_liberation']} after liberation)")


def print_sweep(results: list[dict]):
    """Pretty-print sweep comparison."""
    print(f"\n  {'Config':<25} {'Uniform?':<10} {'Spectral':>10} {'Mix Time':>10} "
          f"{'Conv Mean':>10} {'Conv Med':>10} {'Gap@Lib':>10}")
    print('  ' + '-' * 95)
    for r in results:
        p = r['params']
        label = f"{r['name']} ({p['edge']}/{p['near_edge']}/{p['middle']})"
        print(f"  {label:<25} {'Yes' if r['is_uniform'] else 'NO':<10} "
              f"{r['spectral_gap']:>10.4f} {r['mixing_time']:>10.0f} "
              f"{r['convergence_mean']:>10.0f} {r['convergence_median']:>10.0f} "
              f"{r['gap_at_liberation']:>10.2f}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Markov Chain Analysis Toolkit for Structural Inequality Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --edge 0.10 --near-edge 0.20 --middle 0.25
  %(prog)s --sweep
  %(prog)s --edge 0.10 --near-edge 0.20 --middle 0.25 --barrier-gens 30 --plot
  %(prog)s --edge 0.10 --near-edge 0.20 --middle 0.25 --export results.json
        """,
    )

    # Parameters
    parser.add_argument('--edge', type=float, help='Edge mobility (tiers 1↔2, 6↔7)')
    parser.add_argument('--near-edge', type=float, help='Near-edge mobility (tiers 2↔3, 5↔6)')
    parser.add_argument('--middle', type=float, help='Middle mobility (tiers 3↔4, 4↔5)')

    # Sweep mode
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep over default configs')

    # Scenario options
    parser.add_argument('--barrier-gens', type=int, default=None,
                       help='Generations of discrimination (triggers full simulation)')
    parser.add_argument('--max-gen', type=int, default=300,
                       help='Maximum generations to simulate (default: 300)')
    parser.add_argument('--n-trials', type=int, default=500,
                       help='Monte Carlo trials (default: 500)')
    parser.add_argument('--n-black', type=int, default=50, help='Black population (default: 50)')
    parser.add_argument('--n-white', type=int, default=50, help='White population (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.15,
                       help='Convergence threshold (default: 0.15 = 35-65%% range)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    # Output
    parser.add_argument('--plot', action='store_true', help='Generate visualization charts')
    parser.add_argument('--plot-dir', type=str, default='.', help='Directory for plots')
    parser.add_argument('--export', type=str, default=None,
                       help='Export results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')

    args = parser.parse_args()

    # ---- Sweep mode ----
    if args.sweep:
        print('=' * 60)
        print('PARAMETER SWEEP')
        print('=' * 60)
        barrier = args.barrier_gens or 30
        results = parameter_sweep(barrier_gens=barrier, n_trials=min(args.n_trials, 200))
        print_sweep(results)
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nExported to {args.export}")
        return

    # ---- Single config mode ----
    if args.edge is None or args.near_edge is None or args.middle is None:
        parser.error("Provide --edge, --near-edge, and --middle (or use --sweep)")

    exportable = {'params': {
        'edge': args.edge, 'near_edge': args.near_edge, 'middle': args.middle,
    }}

    # Build and analyze
    T = build_transition_matrix(args.edge, args.near_edge, args.middle)
    analysis = analyze_chain(T)
    exportable['analysis'] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in analysis.items()
    }

    if not args.quiet:
        print('=' * 60)
        print(f'TRANSITION MATRIX — params: {args.edge}/{args.near_edge}/{args.middle}')
        print('=' * 60)
        print_matrix(T)

        print(f'\n{"=" * 60}')
        print('CHAIN ANALYSIS')
        print('=' * 60)
        print_analysis(analysis)

        if not analysis['is_uniform']:
            print('\n  ⚠️  WARNING: Stationary distribution is NOT uniform!')
            print('  This should not happen with the detailed balance construction.')

    # Full scenario
    if args.barrier_gens is not None:
        if not args.quiet:
            print(f'\n{"=" * 60}')
            print(f'FULL SCENARIO — {args.barrier_gens}-gen barrier, {args.n_trials} trials')
            print('=' * 60)

        sim = simulate_scenario(
            T,
            n_black=args.n_black,
            n_white=args.n_white,
            barrier_gens=args.barrier_gens,
            max_gen=args.max_gen,
            n_trials=args.n_trials,
            convergence_threshold=args.threshold,
            seed=args.seed,
        )

        # Don't include huge arrays in export by default
        exportable['scenario'] = {
            k: v for k, v in sim.items()
            if k not in ('avg_gap_trajectory', 'avg_black_distributions', 'avg_white_distributions')
        }
        exportable['scenario']['total_generations'] = len(sim['avg_gap_trajectory'])

        if not args.quiet:
            print_scenario(sim, args.barrier_gens)

        # Plots
        if args.plot:
            print(f'\n{"=" * 60}')
            print('GENERATING PLOTS')
            print('=' * 60)
            generate_plots(
                T, sim,
                barrier_gens=args.barrier_gens,
                params=exportable['params'],
                output_dir=args.plot_dir,
            )

    # Export
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(exportable, f, indent=2, default=str)
        if not args.quiet:
            print(f"\nExported to {args.export}")


if __name__ == '__main__':
    main()
