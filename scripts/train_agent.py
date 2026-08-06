"""
Train Apollo RL betting agent on historical match data.

Data priority:
  1. Club matches (data/processed/matches.parquet) — 32k+ matches with full stats
  2. International results fallback (WC + qualifiers via martj42 CSV)

Usage:
    python scripts/train_agent.py                    # train on club data
    python scripts/train_agent.py --timesteps 100000
    python scripts/train_agent.py --international    # use intl data instead
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.elo_model import EloModel
from core.rl_env import BettingEnv
from core.rl_agent import make_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_URL  = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CLUB_PARQUET = "data/processed/matches.parquet"
MODEL_PATH   = "data/models/elo_national.json"
AGENT_PATH   = "data/models/rl_agent"


def backtest(env: BettingEnv, agent, label: str = ""):
    obs, _ = env.reset()
    bets = wins = 0
    bk_hist = [1.0]
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, _, _ = env.step(action)
        counts[action] += 1
        if action != 0:
            bets += 1
            if reward > 0:
                wins += 1
        bk_hist.append(env._bankroll)
        if done:
            break

    final  = env._bankroll
    roi    = (final - 1.0) * 100
    wr     = (wins / bets * 100) if bets else 0
    arr    = np.array(bk_hist)
    peak   = np.maximum.accumulate(arr)
    max_dd = float(((peak - arr) / peak).max() * 100)

    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Matches   : {len(env.df)}")
    print(f"  Bets      : {bets} ({bets/len(env.df)*100:.1f}%)")
    print(f"  Win rate  : {wr:.1f}%")
    print(f"  ROI       : {roi:+.2f}%")
    print(f"  Bankroll  : {final:.4f}")
    print(f"  Max DD    : {max_dd:.2f}%")
    print(f"  Actions   : no_bet={counts[0]} home={counts[1]} draw={counts[2]} away={counts[3]}")
    return {"roi": roi, "bets": bets, "win_rate": wr, "bankroll": final, "max_dd": max_dd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",    type=int,  default=50_000)
    parser.add_argument("--international", action="store_true", help="Use international data instead of club")
    parser.add_argument("--model",        default=MODEL_PATH)
    parser.add_argument("--agent-out",    default=AGENT_PATH)
    args = parser.parse_args()

    # Load Elo model (needed for international path; also used for club Elo features)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Elo model not found. Run: python scripts/build_elo.py")
        sys.exit(1)
    elo_model = EloModel.load(str(model_path))

    # Build environment
    if args.international:
        logger.info("Building env from international results CSV...")
        env = BettingEnv.from_international_csv(RESULTS_URL, elo_model)
    else:
        club_path = Path(CLUB_PARQUET)
        if not club_path.exists():
            print(f"Club data not found at {club_path}. Run: python -m core.data_loader download && python -m core.data_loader process")
            sys.exit(1)
        logger.info("Building env from club matches: %s", club_path)
        env = BettingEnv.from_processed_data(str(club_path))

    logger.info("Total matches in env: %d", len(env.df))
    if len(env.df) < 100:
        print("Too few matches to train meaningfully.")
        sys.exit(1)

    # Chronological 80/20 split — no leakage
    split     = int(len(env.df) * 0.8)
    train_df  = env.df.iloc[:split].reset_index(drop=True)
    eval_df   = env.df.iloc[split:].reset_index(drop=True)
    train_env = BettingEnv(train_df)
    eval_env  = BettingEnv(eval_df)

    logger.info("Train: %d matches | Eval: %d matches", len(train_df), len(eval_df))

    # Build + train agent
    agent, agent_type = make_agent(train_env, use_dqn=True)
    logger.info("Agent type: %s", agent_type)

    if agent_type == "dqn":
        logger.info("Training DQN for %d timesteps...", args.timesteps)
        agent.learn(total_timesteps=args.timesteps, progress_bar=True)
    else:
        logger.info("Training QTable agent (200 episodes)...")
        for ep in range(200):
            obs, _ = train_env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs)
                next_obs, reward, done, _, _ = train_env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
            if ep % 50 == 0:
                logger.info("  ep=%d epsilon=%.3f", ep, agent.epsilon)

    # Save
    out = Path(args.agent_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if agent_type == "dqn":
        agent.save(str(out) + "_dqn")
        print(f"\nSaved DQN -> {out}_dqn.zip")
    else:
        agent.save(str(out) + ".json")
        print(f"\nSaved QTable -> {out}.json")

    # Backtest both splits
    backtest(train_env, agent, "Train set")
    backtest(eval_env,  agent, "Eval set (holdout)")


if __name__ == "__main__":
    main()
