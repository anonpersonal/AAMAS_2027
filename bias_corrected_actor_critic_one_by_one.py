import argparse
import os
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise


def estimate_q_values(model, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return all critic estimates for Q(s, a)."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).to(model.device)

    with torch.no_grad():
        q_values = model.critic(obs_tensor, action_tensor)

    return np.asarray(
        [float(q_value.item()) for q_value in q_values],
        dtype=np.float32,
    )


def compute_reference_return(
    model,
    eval_env,
    obs: np.ndarray,
    gamma: float,
    rollout_steps: int = 1000,
    episodes_per_state: int = 5,
) -> float:
    """Estimate the discounted MC return from a replay-buffer state for Pendulum-v1."""
    theta = float(np.arctan2(obs[1], obs[0]))
    theta_dot = float(obs[2])

    base_env = eval_env.unwrapped
    episode_returns = []

    for _ in range(episodes_per_state):
        eval_env.reset()
        base_env.state = np.array([theta, theta_dot], dtype=np.float32)

        current_obs = np.array(
            [np.cos(theta), np.sin(theta), theta_dot],
            dtype=np.float32,
        )

        discounted_return = 0.0
        discount = 1.0

        for _ in range(rollout_steps):
            action, _ = model.predict(current_obs, deterministic=True)
            current_obs, reward, terminated, truncated, _ = eval_env.step(action)

            discounted_return += discount * float(reward)
            discount *= gamma

            if terminated or truncated:
                break

        episode_returns.append(discounted_return)

    return float(np.mean(episode_returns))


def evaluate_buffer_bias(
    model,
    eval_env,
    gamma: float,
    num_states: int = 20,
    episodes_per_state: int = 5,
    tau: float = 5.0,
) -> Optional[dict]:
    """Compare critic estimates with Monte Carlo reference returns and compute GPR bias correction."""
    if not hasattr(model, "replay_buffer"):
        return None

    buffer_size = model.replay_buffer.size()
    if buffer_size == 0:
        return None

    sample_size = min(num_states, buffer_size)
    buffer_data = model.replay_buffer.sample(sample_size)
    obs_samples = buffer_data.observations.cpu().numpy()

    q1_predictions = []
    q2_predictions = []
    q_min_predictions = []
    reference_returns = []
    actions_list = []

    for obs in obs_samples:
        obs = obs.flatten()
        action, _ = model.predict(obs, deterministic=True)
        q_values = estimate_q_values(model, obs, action)

        reference_return = compute_reference_return(
            model=model,
            eval_env=eval_env,
            obs=obs,
            gamma=gamma,
            rollout_steps=1000,
            episodes_per_state=episodes_per_state,
        )

        q1_predictions.append(q_values[0])
        q2_predictions.append(q_values[1] if len(q_values) > 1 else np.nan)
        q_min_predictions.append(np.min(q_values))
        reference_returns.append(reference_return)
        actions_list.append(action.flatten())

    q1_arr = np.array(q1_predictions, dtype=np.float32)
    q_min_arr = np.array(q_min_predictions, dtype=np.float32)
    ref_arr = np.array(reference_returns, dtype=np.float32)
    obs_arr = np.array([o.flatten() for o in obs_samples], dtype=np.float32)
    act_arr = np.array(actions_list, dtype=np.float32)

    # --- Gaussian Process Regression (GPR) Bias Corrector ---
    # Input features: x = [s, a]
    X_features = np.hstack([obs_arr, act_arr])
    # Target anchor error: e = Q_critic - Q_MC
    anchor_errors = q_min_arr - ref_arr

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=1,
        random_state=42,
    )

    try:
        gpr.fit(X_features, anchor_errors)
        mu_e, sigma_e = gpr.predict(X_features, return_std=True)
    except Exception:
        mu_e = anchor_errors
        sigma_e = np.zeros_like(anchor_errors)

    # Uncertainty-gated scaling factor lambda(x) = exp(- sigma_e^2 / (2 * tau^2))
    lambda_weights = np.exp(- (sigma_e ** 2) / (2.0 * (tau ** 2)))
    q_corrected_arr = q_min_arr - (lambda_weights * mu_e)
    gpr_corrected_bias_arr = q_corrected_arr - ref_arr

    q2_array = np.asarray(q2_predictions, dtype=np.float32)

    return {
        "q1": float(np.mean(q1_arr)),
        "q2": (
            float(np.mean(q2_array[np.isfinite(q2_array)]))
            if np.isfinite(q2_array).any()
            else None
        ),
        "q_min": float(np.mean(q_min_arr)),
        "reference_return": float(np.mean(ref_arr)),
        "uncorrected_bias": float(np.mean(anchor_errors)),
        "gpr_q_corrected": float(np.mean(q_corrected_arr)),
        "gpr_corrected_bias": float(np.mean(gpr_corrected_bias_arr)),
    }



# -------------------------------------------------------------
# Old episode-based research protocol, run ONE algorithm/seed.
# The evaluate_buffer_bias(...) function above is copied from
# the user's original code WITHOUT altering its methodology.
# -------------------------------------------------------------
import json
import shutil
import subprocess
import sys
from pathlib import Path

CSV_COLS = [
    "algorithm", "seed", "episode", "q1_estimate", "q2_estimate",
    "q_min_estimate", "reference_return", "raw_bias", "gpr_corrected_bias"
]


def write_csv_atomic(rows, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    pd.DataFrame(rows, columns=CSV_COLS).to_csv(temporary, index=False)
    os.replace(temporary, target)


def save_checkpoint(model, directory, completed_episodes, config, history):
    """Commit model, replay, and metrics together in an atomic directory rename."""
    directory = Path(directory)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    scratch = parent / (".incomplete_" + directory.name)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    try:
        # SB3 saves the policy/optimizer state and the replay buffer separately.
        model.save(str(scratch / "model"))
        model.save_replay_buffer(str(scratch / "replay_buffer.pkl"))
        manifest = {
            "completed_episodes": int(completed_episodes),
            "model_num_timesteps": int(model.num_timesteps),
            "configuration": config,
            "history": history,
        }
        (scratch / "state.json").write_text(json.dumps(manifest, indent=2))
        if directory.exists():
            raise FileExistsError(f"Checkpoint already exists: {directory}")
        os.rename(scratch, directory)
    except BaseException:
        # Leave any interrupted temporary directory for later inspection,
        # but never treat it as a committed checkpoint.
        raise

    # CSV is a convenient export; checkpoint JSON is the authoritative history.
    write_csv_atomic(history, parent.parent / "eval_metrics.csv")
    # Limit replay-buffer disk use: keep the 2 newest complete checkpoints.
    complete = sorted(parent.glob("ckpt_ep_*"))
    for old in complete[:-2]:
        if old != directory:
            shutil.rmtree(old)
    print(f"CHECKPOINT SAVED: {directory} | episodes={completed_episodes}", flush=True)


def find_resume_checkpoint(seed_dir, expected_config):
    checkpoints = sorted((seed_dir / "checkpoints").glob("ckpt_ep_*"), reverse=True)
    for cp in checkpoints:
        try:
            manifest = json.loads((cp / "state.json").read_text())
            if manifest["configuration"] != expected_config:
                raise ValueError("Configuration differs from this run. Stop rather than mixing protocols.")
            if not (cp / "model.zip").is_file() or not (cp / "replay_buffer.pkl").is_file():
                print(f"Skipping incomplete checkpoint: {cp}", flush=True)
                continue
            return cp, manifest
        except (OSError, json.JSONDecodeError, KeyError) as err:
            print(f"Skipping invalid checkpoint {cp}: {err}", flush=True)
    return None, None


class SafeEpisodeEvaluationCallback(BaseCallback):
    def __init__(self, eval_env, config, seed_dir, start_episode=0, history=None):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.config = config
        self.seed_dir = Path(seed_dir)
        self.completed_episodes = start_episode
        self.history = list(history or [])
        self.pending_checkpoint = False

    def _on_step(self):
        # One Gymnasium Pendulum-v1 episode is 200 steps; the callback counts
        # actual completed episodes rather than inferring episodes from steps.
        dones = self.locals.get("dones")
        if dones is None:
            dones = self.locals.get("done")
        if dones is None:
            return True
        for done in np.asarray(dones, dtype=bool).reshape(-1):
            if not done:
                continue
            self.completed_episodes += 1
            is_final = self.completed_episodes >= self.config["total_training_episodes"]
            if self.completed_episodes % self.config["eval_every_episodes"] == 0 or is_final:
                metrics = evaluate_buffer_bias(
                    model=self.model,
                    eval_env=self.eval_env,
                    gamma=self.config["gamma"],
                    num_states=self.config["num_states"],
                    episodes_per_state=self.config["episodes_per_state"],
                    tau=self.config["tau"],
                )
                if metrics is None:
                    raise RuntimeError("Replay buffer was empty at evaluation checkpoint")
                row = {
                    "algorithm": self.config["algorithm"],
                    "seed": self.config["seed"],
                    "episode": self.completed_episodes,
                    "q1_estimate": metrics["q1"],
                    "q2_estimate": metrics["q2"],
                    "q_min_estimate": metrics["q_min"],
                    "reference_return": metrics["reference_return"],
                    "raw_bias": metrics["uncorrected_bias"],
                    "gpr_corrected_bias": metrics["gpr_corrected_bias"],
                }
                self.history.append(row)
                # Persist CSV immediately: plotting remains a separate, manual step.
                # The next committed checkpoint holds authoritative resume history.
                write_csv_atomic(self.history, self.seed_dir / "eval_metrics.csv")
                print(
                    f"Episode {self.completed_episodes} | Qmin: {metrics['q_min']:.2f} | "
                    f"Ref Return: {metrics['reference_return']:.2f} | "
                    f"Raw Bias: {metrics['uncorrected_bias']:.2f} | "
                    f"GPR Bias: {metrics['gpr_corrected_bias']:.2f}", flush=True
                )
                # Save after SB3 stores this transition in the replay buffer.
                self.pending_checkpoint = True
        return True

    def _on_rollout_end(self):
        if self.pending_checkpoint:
            save_checkpoint(
                self.model,
                self.seed_dir / "checkpoints" / f"ckpt_ep_{self.completed_episodes:06d}",
                self.completed_episodes, self.config, self.history,
            )
            self.pending_checkpoint = False


def parse_one_run_args():
    p = argparse.ArgumentParser(
        description="Old GPR bias experiment: run all 5 seeds sequentially for one algorithm"
    )
    p.add_argument("--algorithm", choices=["DDPG", "TD3", "SAC"], required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--seed", type=int, help="Run an individual seed (worker mode)")
    selection.add_argument("--seeds", type=str,
                           help="Run seeds sequentially, e.g. 0,1,2,3,4 (batch mode)")
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--total_training_episodes", type=int, default=10000)
    p.add_argument("--eval_every_episodes", type=int, default=200)
    p.add_argument("--num_states", type=int, default=50)
    p.add_argument("--episodes_per_state", type=int, default=10)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--output_dir", default="results_old_one_by_one")
    p.add_argument("--resume", action="store_true", help="Continue from the newest checkpoint if present")
    return p.parse_args()


def main_one_run(args):
    if args.env not in {"Ant-v4", "Ant-v5"}:
        raise ValueError("This version supports Ant-v4 and Ant-v5 only.")
    if args.total_training_episodes <= 0 or args.eval_every_episodes <= 0:
        raise ValueError("Training/evaluation episode counts must be positive")
    if args.num_states <= 0 or args.episodes_per_state <= 0:
        raise ValueError("Evaluation sample counts must be positive")
    if args.seed < 0:
        raise ValueError("Use a nonnegative seed")
    seed_dir = Path(args.output_dir) / f"{args.algorithm}_seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "env": args.env,
        "total_training_episodes": args.total_training_episodes,
        "eval_every_episodes": args.eval_every_episodes,
        "num_states": args.num_states,
        "episodes_per_state": args.episodes_per_state,
        "tau": args.tau,
        "gamma": 0.99,
    }

    train_env = gym.make(args.env)
    eval_env = gym.make(args.env)
    try:
        #max_steps = getattr(train_env.spec, "max_episode_steps", None) or 200
        #if max_steps != 200:
        #   raise ValueError("Expected Pendulum-v1 200-step episodes")
        max_steps = getattr(train_env.spec, "max_episode_steps", None)
        if max_steps != 1000:
            raise ValueError(
                f"Expected Ant-v5 with 1000-step episodes, got {max_steps}"
            )
        
        model_class = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC}[args.algorithm]
        cp = None
        state = None
        if args.resume:
            cp, state = find_resume_checkpoint(seed_dir, config)
        elif list((seed_dir / "checkpoints").glob("ckpt_ep_*")):
            raise RuntimeError("Checkpoints already exist. Use --resume or a different output_dir.")
        start_ep = 0
        history = []
        if cp is not None:
            model = model_class.load(str(cp / "model.zip"), env=train_env, device="auto")
            model.load_replay_buffer(str(cp / "replay_buffer.pkl"))
            start_ep = int(state["completed_episodes"])
            history = state["history"]
            if model.num_timesteps != int(state["model_num_timesteps"]):
                raise RuntimeError("Timesteps mismatch between model checkpoint and history")
            if model.num_timesteps != start_ep * max_steps:
                raise RuntimeError("Timesteps mismatch with expected fixed Pendulum episode length")
            print(f"RESUMED {args.algorithm} seed={args.seed} at episode={start_ep} "
                  f"(timesteps={model.num_timesteps})", flush=True)
            write_csv_atomic(history, seed_dir / "eval_metrics.csv")
        else:
            existing_csv = seed_dir / "eval_metrics.csv"
            if existing_csv.exists() and existing_csv.stat().st_size > 0:
                raise RuntimeError(
                    f"Existing CSV without a compatible checkpoint: {existing_csv}. "
                    "Do not overwrite old results. Recover a matching checkpoint or "
                    "choose a NEW --output_dir."
                )
            model_kwargs = {"gamma": 0.99, "seed": args.seed, "verbose": 0}
            if args.algorithm in {"DDPG", "TD3"}:
                n_actions = train_env.action_space.shape[0]
                model_kwargs["action_noise"] = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
            if args.algorithm == "TD3":
                model_kwargs.update(policy_delay=2, target_policy_noise=0.2,
                                    target_noise_clip=0.5)
            if args.algorithm == "SAC":
                model_kwargs["ent_coef"] = "auto"
            model = model_class(policy="MlpPolicy", env=train_env, **model_kwargs)
            print(f"STARTING NEW {args.algorithm} seed={args.seed}", flush=True)
        if start_ep >= args.total_training_episodes:
            print(f"ALREADY COMPLETE: {args.algorithm} seed={args.seed}; no training needed.", flush=True)
            return
        callback = SafeEpisodeEvaluationCallback(eval_env, config, seed_dir, start_ep, history)
        remaining = args.total_training_episodes - start_ep
        print(f"Target episodes={args.total_training_episodes}; remaining={remaining}; "
              f"evaluation every {args.eval_every_episodes}; "
              f"num_states={args.num_states}; MC rollouts per state={args.episodes_per_state}", flush=True)
        print("WARNING: checkpoints permit practical continuation, not bit-for-bit identical reproduction.", flush=True)
        model.learn(total_timesteps=remaining * max_steps, callback=callback,
                    reset_num_timesteps=(start_ep == 0))
        if callback.completed_episodes != args.total_training_episodes:
            raise RuntimeError(f"Stopped at {callback.completed_episodes} episodes, "
                               f"expected {args.total_training_episodes}")
        write_csv_atomic(callback.history, seed_dir / "eval_metrics.csv")
        print(f"FINISHED {args.algorithm} seed={args.seed} "
              f"| {len(callback.history)} evaluation rows | "
              f"{seed_dir / 'eval_metrics.csv'}", flush=True)
    finally:
        train_env.close()
        eval_env.close()


def verify_finished_seed(seed_dir, expected_algo, expected_seed, target, eval_interval):
    """Refuse to mark a seed finished unless every required evaluation exists."""
    csv_file = seed_dir / "eval_metrics.csv"
    if not csv_file.is_file():
        raise RuntimeError(f"Expected CSV was not saved: {csv_file}")
    df = pd.read_csv(csv_file)
    expected_episodes = list(range(eval_interval, target + 1, eval_interval))
    if not expected_episodes or expected_episodes[-1] != target:
        expected_episodes.append(target)
    if not {"algorithm", "seed", "episode"}.issubset(df.columns):
        raise RuntimeError(f"CSV is missing required columns: {csv_file}")
    if (df["algorithm"] != expected_algo).any() or (df["seed"] != expected_seed).any():
        raise RuntimeError(f"CSV algorithm/seed mismatch in {csv_file}")
    episodes = pd.to_numeric(df["episode"], errors="raise").tolist()
    if episodes != expected_episodes:
        raise RuntimeError(
            f"Incomplete or unexpected evaluations for {expected_algo} seed {expected_seed}. "
            f"Found {len(episodes)} rows; expected {len(expected_episodes)}. "
            "Do not proceed without checking the saved experiment."
        )
    print(f"VERIFIED {expected_algo} seed {expected_seed}: {len(episodes)} evaluations", flush=True)


def run_algorithm_batch(args):
    """Run each seed in its own subprocess to release GPU memory between seeds."""
    try:
        seeds = [int(x.strip()) for x in args.seeds.split(",")]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated nonnegative integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        raise ValueError("--seeds must contain distinct, nonnegative seed numbers")
    print(f"ALGORITHM: {args.algorithm} | SEEDS: {seeds} (sequential, ONE GPU)", flush=True)
    script = str(Path(__file__).resolve())
    for index, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 70}\nSTART {args.algorithm} seed {seed} "
              f"({index}/{len(seeds)})\n{'=' * 70}", flush=True)
        command = [
            sys.executable, "-u", script,
            "--algorithm", args.algorithm,
            "--seed", str(seed),
            "--env", args.env,
            "--total_training_episodes", str(args.total_training_episodes),
            "--eval_every_episodes", str(args.eval_every_episodes),
            "--num_states", str(args.num_states),
            "--episodes_per_state", str(args.episodes_per_state),
            "--tau", str(args.tau),
            "--output_dir", args.output_dir,
        ]
        if args.resume:
            command.append("--resume")
        # A failure stops the batch: never silently skip an incomplete seed.
        subprocess.run(command, check=True)
        seed_dir = Path(args.output_dir) / f"{args.algorithm}_seed{seed}"
        verify_finished_seed(seed_dir, args.algorithm, seed,
                             args.total_training_episodes, args.eval_every_episodes)
    print(f"\nALL REQUESTED SEEDS COMPLETE FOR {args.algorithm}.", flush=True)
    print("Plots are NOT generated during training.", flush=True)


if __name__ == "__main__":
    arguments = parse_one_run_args()
    if arguments.seeds is not None:
        run_algorithm_batch(arguments)
    else:
        main_one_run(arguments)
