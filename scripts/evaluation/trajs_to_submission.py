#!/usr/bin/env python3
"""Convert trajectories.pkl (from eval --save_trajectories) into a submission.pkl.

trajectories.pkl: {token: np.ndarray (8, 3)}
submission.pkl:    team metadata + {"predictions": [{token: Trajectory}]}

Example:
    python scripts/evaluation/trajs_to_submission.py \
        --trajectories /path/to/trajectories.pkl \
        --team_name "My Team" --authors "Me" --email me@x.com \
        --institution "X" --country "Y"
"""
import argparse
import pickle

from navsim.common.dataclasses import Trajectory


def main():
    p = argparse.ArgumentParser(description="trajectories.pkl -> submission.pkl")
    p.add_argument("--trajectories", required=True, help="Path to trajectories.pkl")
    p.add_argument("--output", help="Output path (default: alongside input)")
    p.add_argument("--team_name", default="")
    p.add_argument("--authors", default="")
    p.add_argument("--email", default="")
    p.add_argument("--institution", default="")
    p.add_argument("--country", default="")
    args = p.parse_args()

    with open(args.trajectories, "rb") as f:
        trajs = pickle.load(f)

    output = {token: Trajectory(traj) for token, traj in trajs.items()}
    submission = {
        "team_name": args.team_name,
        "authors": args.authors,
        "email": args.email,
        "institution": args.institution,
        "country / region": args.country,
        "predictions": [output],
    }

    out_path = args.output or args.trajectories.replace(".pkl", "_submission.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(submission, f)
    print(f"Wrote {len(output)} predictions to {out_path}")


if __name__ == "__main__":
    main()
