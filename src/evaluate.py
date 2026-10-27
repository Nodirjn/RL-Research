"""Evaluate a saved PPO model on fixed seeds (deterministic policy unless --stochastic).

Usage:
    python src/evaluate.py --run-dir experiments/EXP-001/C2_cleanrl_3M_s0 [--model final|1000000]
        [--episodes 50] [--seed-offset 20000] [--stochastic] [--out report.json]

Uses the same `run_episodes` code as training, with the VecNormalize statistics saved
next to the model (frozen), so results are directly comparable to eval.csv/final_eval.json.
"""
import argparse
import json

from common import load_policy, run_episodes, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="final", help="'final' or a checkpoint step, e.g. 1000000")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=20000)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mean/argmax")
    parser.add_argument("--out", help="write the report as JSON to this path")
    args = parser.parse_args()

    model, normalize_obs, cfg = load_policy(args.run_dir, args.model)
    episodes = run_episodes(model, cfg, args.episodes, args.seed_offset, normalize_obs, not args.stochastic)
    report = {"run_dir": args.run_dir, "model": args.model, "timesteps": int(model.num_timesteps),
              "seed_offset": args.seed_offset, "deterministic": not args.stochastic, **summarize(episodes),
              "episodes": episodes}
    rates = " ".join(f"{k}={report[k]:.2f}" for k in ("fall_rate", "mean_x_displacement", "mean_x_velocity",
                                                       "success_rate", "death_rate", "timeout_rate") if k in report)
    print(f"{args.run_dir} [{args.model}] steps={report['timesteps']} episodes={args.episodes} "
          f"{'stochastic' if args.stochastic else 'deterministic'}: "
          f"return={report['mean_return']:.2f}+-{report['std_return']:.2f} len={report['mean_length']:.0f} {rates}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
