#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARL Experiment Agent
=====================

A single-file, runnable multi-agent assistant for multi-agent reinforcement learning experiments.
It helps with four repetitive tasks:

1. Parse and validate experiment configuration.
2. Generate reproducible training commands/scripts for multiple algorithms, scenarios, and seeds.
3. Collect CSV/TensorBoard-like scalar logs and summarize final performance.
4. Produce a Markdown experiment report with warnings, tables, and suggested next actions.

Design principle:
- This is not tied to one MARL codebase. It assumes your training entrypoint can receive CLI args.
- It can be adapted to MAPPO/HAPPO/MADDPG/HASAC/HARL/on-policy style repositories.
- It uses only Python standard libraries by default. PyYAML and TensorBoard are optional.

Example usage:

    # 1) Create a template config
    python marl_experiment_agent.py init --out experiment_config.json

    # 2) Validate the config
    python marl_experiment_agent.py validate --config experiment_config.json

    # 3) Generate runnable scripts
    python marl_experiment_agent.py plan --config experiment_config.json --outdir agent_outputs

    # 4) Analyze finished experiment logs
    python marl_experiment_agent.py analyze --config experiment_config.json --logdir runs --outdir agent_outputs

    # 5) Do all non-training steps
    python marl_experiment_agent.py all --config experiment_config.json --logdir runs --outdir agent_outputs

Expected CSV log format:
    step,episode_reward,capture_rate,episode_length,success_rate
    10000,12.3,0.18,100,0.18
    20000,15.8,0.22,100,0.22

If your logs use different column names, modify metric_aliases in the config.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import glob
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def std(xs: Sequence[float]) -> Optional[float]:
    if len(xs) <= 1:
        return 0.0 if xs else None
    return float(statistics.stdev(xs))


def format_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def quote_arg(value: Any) -> str:
    return shlex.quote(str(value))


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def load_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = read_text(path)
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".toml"}:
        try:
            import tomllib  # Python 3.11+
        except Exception as exc:
            raise RuntimeError("TOML requires Python 3.11+ tomllib, or use JSON config.") from exc
        return tomllib.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml, or use JSON.") from exc
        data = yaml.safe_load(text)
        return data or {}
    raise ValueError(f"Unsupported config file format: {path.suffix}. Use .json, .toml, .yaml, or .yml")


def save_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# Data schema
# -----------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    env_args: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"


@dataclass
class TrainingDefaults:
    entrypoint: str = "train.py"
    python_bin: str = "python"
    common_args: Dict[str, Any] = field(default_factory=dict)
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    conda_env: Optional[str] = None
    shell_prefix: str = ""
    dry_run: bool = True


@dataclass
class ExperimentConfig:
    project_name: str
    algorithms: List[str]
    scenarios: List[Scenario]
    seeds: List[int]
    training: TrainingDefaults
    algorithm_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_aliases: Dict[str, List[str]] = field(default_factory=dict)
    primary_metric: str = "capture_rate"
    higher_is_better: bool = True
    last_n_points: int = 10
    min_expected_runs: int = 1

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ExperimentConfig":
        scenarios_raw = raw.get("scenarios", [])
        scenarios: List[Scenario] = []
        for s in scenarios_raw:
            if isinstance(s, str):
                scenarios.append(Scenario(name=s))
            elif isinstance(s, dict):
                scenarios.append(
                    Scenario(
                        name=str(s.get("name", "scenario")),
                        env_args=dict(s.get("env_args", {})),
                        difficulty=str(s.get("difficulty", "medium")),
                    )
                )
            else:
                raise ValueError(f"Invalid scenario item: {s}")

        tr_raw = dict(raw.get("training", {}))
        training = TrainingDefaults(
            entrypoint=str(tr_raw.get("entrypoint", "train.py")),
            python_bin=str(tr_raw.get("python_bin", "python")),
            common_args=dict(tr_raw.get("common_args", {})),
            gpu_ids=[int(x) for x in tr_raw.get("gpu_ids", [0])],
            conda_env=tr_raw.get("conda_env"),
            shell_prefix=str(tr_raw.get("shell_prefix", "")),
            dry_run=bool(tr_raw.get("dry_run", True)),
        )

        return ExperimentConfig(
            project_name=str(raw.get("project_name", "marl_project")),
            algorithms=[str(x) for x in raw.get("algorithms", [])],
            scenarios=scenarios,
            seeds=[int(x) for x in raw.get("seeds", [1])],
            training=training,
            algorithm_args={str(k): dict(v) for k, v in raw.get("algorithm_args", {}).items()},
            metric_aliases={str(k): [str(x) for x in v] for k, v in raw.get("metric_aliases", {}).items()},
            primary_metric=str(raw.get("primary_metric", "capture_rate")),
            higher_is_better=bool(raw.get("higher_is_better", True)),
            last_n_points=int(raw.get("last_n_points", 10)),
            min_expected_runs=int(raw.get("min_expected_runs", 1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "multi_uav_pursuit_experiment",
    "algorithms": ["MAPPO", "HAPPO", "MADDPG", "HASAC"],
    "scenarios": [
        {
            "name": "Easy_3v1_3obs",
            "difficulty": "easy",
            "env_args": {
                "num_pursuers": 3,
                "num_evaders": 1,
                "num_obstacles": 3,
                "max_cycles": 100,
                "capture_radius": 0.04,
            },
        },
        {
            "name": "Medium_4v2_5obs",
            "difficulty": "medium",
            "env_args": {
                "num_pursuers": 4,
                "num_evaders": 2,
                "num_obstacles": 5,
                "max_cycles": 100,
                "capture_radius": 0.04,
            },
        },
        {
            "name": "Hard_5v3_8obs",
            "difficulty": "hard",
            "env_args": {
                "num_pursuers": 5,
                "num_evaders": 3,
                "num_obstacles": 8,
                "max_cycles": 120,
                "capture_radius": 0.04,
            },
        },
    ],
    "seeds": [1, 2, 3],
    "training": {
        "entrypoint": "train.py",
        "python_bin": "python",
        "gpu_ids": [0],
        "conda_env": None,
        "shell_prefix": "",
        "dry_run": True,
        "common_args": {
            "env_name": "MultiUAVPursuit",
            "total_timesteps": 1000000,
            "episode_length": 100,
            "use_wandb": False,
            "log_interval": 10,
            "eval_interval": 50,
        },
    },
    "algorithm_args": {
        "MAPPO": {"lr": 0.0005, "ppo_epoch": 10, "num_mini_batch": 2},
        "HAPPO": {"lr": 0.0005, "ppo_epoch": 10, "num_mini_batch": 2},
        "MADDPG": {"actor_lr": 0.0001, "critic_lr": 0.001, "buffer_size": 100000},
        "HASAC": {"actor_lr": 0.0003, "critic_lr": 0.0003, "alpha_lr": 0.0003},
    },
    "metric_aliases": {
        "step": ["step", "global_step", "timestep", "timesteps", "env_step"],
        "episode_reward": ["episode_reward", "reward", "average_episode_rewards", "eval_average_episode_rewards"],
        "capture_rate": ["capture_rate", "eval_capture_rate", "success_rate", "win_rate"],
        "episode_length": ["episode_length", "eval_episode_length"],
        "loss": ["loss", "value_loss", "policy_loss"],
    },
    "primary_metric": "capture_rate",
    "higher_is_better": True,
    "last_n_points": 10,
    "min_expected_runs": 1,
}


# -----------------------------------------------------------------------------
# Agent base
# -----------------------------------------------------------------------------


@dataclass
class AgentMessage:
    source: str
    level: str
    message: str


class BaseAgent:
    name = "BaseAgent"

    def __init__(self) -> None:
        self.messages: List[AgentMessage] = []

    def info(self, msg: str) -> None:
        self.messages.append(AgentMessage(self.name, "INFO", msg))

    def warn(self, msg: str) -> None:
        self.messages.append(AgentMessage(self.name, "WARN", msg))

    def error(self, msg: str) -> None:
        self.messages.append(AgentMessage(self.name, "ERROR", msg))


# -----------------------------------------------------------------------------
# Config Agent
# -----------------------------------------------------------------------------


class ConfigAgent(BaseAgent):
    name = "ConfigAgent"

    REQUIRED_ALGOS = {"MAPPO", "HAPPO", "MADDPG", "HASAC", "QMIX", "VDN", "IPPO", "MAT", "COMA"}

    def load(self, config_path: Path) -> ExperimentConfig:
        raw = load_config_file(config_path)
        cfg = ExperimentConfig.from_dict(raw)
        self.info(f"Loaded config from {config_path}")
        return cfg

    def validate(self, cfg: ExperimentConfig) -> List[AgentMessage]:
        if not cfg.algorithms:
            self.error("No algorithms configured.")
        if not cfg.scenarios:
            self.error("No scenarios configured.")
        if not cfg.seeds:
            self.error("No seeds configured.")
        if not cfg.training.entrypoint:
            self.error("training.entrypoint is empty.")
        if cfg.last_n_points <= 0:
            self.error("last_n_points must be positive.")
        if cfg.min_expected_runs <= 0:
            self.error("min_expected_runs must be positive.")

        unknown_algos = [a for a in cfg.algorithms if a.upper() not in self.REQUIRED_ALGOS]
        if unknown_algos:
            self.warn(
                "Some algorithm names are not in the built-in known list: "
                + ", ".join(unknown_algos)
                + ". This is allowed, but check your train.py argument mapping."
            )

        scenario_names = [s.name for s in cfg.scenarios]
        if len(scenario_names) != len(set(scenario_names)):
            self.error("Scenario names must be unique.")

        for s in cfg.scenarios:
            for key in ["num_pursuers", "num_evaders", "num_obstacles"]:
                if key in s.env_args and int(s.env_args[key]) < 0:
                    self.error(f"Scenario {s.name}: {key} must be non-negative.")
            if "max_cycles" in s.env_args and int(s.env_args["max_cycles"]) <= 0:
                self.error(f"Scenario {s.name}: max_cycles must be positive.")

        if not cfg.metric_aliases.get(cfg.primary_metric):
            self.warn(
                f"primary_metric '{cfg.primary_metric}' has no aliases. "
                "CSV analysis will only match the exact column name."
            )

        self.info("Config validation finished.")
        return self.messages


# -----------------------------------------------------------------------------
# Script Agent
# -----------------------------------------------------------------------------


@dataclass
class RunSpec:
    algorithm: str
    scenario: str
    seed: int
    gpu_id: int
    run_name: str
    output_dir: Path
    command: List[str]


class ScriptAgent(BaseAgent):
    name = "ScriptAgent"

    def build_run_specs(self, cfg: ExperimentConfig, outdir: Path) -> List[RunSpec]:
        run_specs: List[RunSpec] = []
        gpu_ids = cfg.training.gpu_ids or [0]
        i = 0
        for algo in cfg.algorithms:
            algo_args = cfg.algorithm_args.get(algo, {})
            for scenario in cfg.scenarios:
                for seed in cfg.seeds:
                    gpu_id = gpu_ids[i % len(gpu_ids)]
                    run_name = slugify(f"{cfg.project_name}_{algo}_{scenario.name}_seed{seed}")
                    output_dir = outdir / "runs" / run_name
                    args: Dict[str, Any] = {}
                    args.update(cfg.training.common_args)
                    args.update(algo_args)
                    args.update(scenario.env_args)
                    args.update({
                        "algorithm": algo,
                        "scenario_name": scenario.name,
                        "seed": seed,
                        "run_name": run_name,
                        "log_dir": str(output_dir),
                    })
                    command = [cfg.training.python_bin, cfg.training.entrypoint]
                    for k, v in args.items():
                        command.extend(self._arg_to_cli(k, v))
                    run_specs.append(
                        RunSpec(
                            algorithm=algo,
                            scenario=scenario.name,
                            seed=seed,
                            gpu_id=gpu_id,
                            run_name=run_name,
                            output_dir=output_dir,
                            command=command,
                        )
                    )
                    i += 1
        self.info(f"Built {len(run_specs)} run specifications.")
        return run_specs

    @staticmethod
    def _arg_to_cli(key: str, value: Any) -> List[str]:
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            return [flag, str(value).lower()]
        if isinstance(value, (list, tuple)):
            return [flag] + [str(x) for x in value]
        if value is None:
            return []
        return [flag, str(value)]

    def command_to_shell(self, spec: RunSpec, cfg: ExperimentConfig) -> str:
        env_prefix = f"CUDA_VISIBLE_DEVICES={spec.gpu_id} "
        cmd = " ".join(quote_arg(x) for x in spec.command)
        parts: List[str] = []
        if cfg.training.shell_prefix.strip():
            parts.append(cfg.training.shell_prefix.strip())
        if cfg.training.conda_env:
            parts.append(f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {quote_arg(cfg.training.conda_env)}")
        parts.append(env_prefix + cmd)
        return " && ".join(parts)

    def generate_scripts(self, cfg: ExperimentConfig, outdir: Path) -> Dict[str, Path]:
        ensure_dir(outdir)
        specs = self.build_run_specs(cfg, outdir)

        manifest = [
            {
                "algorithm": s.algorithm,
                "scenario": s.scenario,
                "seed": s.seed,
                "gpu_id": s.gpu_id,
                "run_name": s.run_name,
                "output_dir": str(s.output_dir),
                "command": s.command,
                "shell_command": self.command_to_shell(s, cfg),
            }
            for s in specs
        ]
        manifest_path = outdir / "run_manifest.json"
        save_json(manifest_path, manifest)

        bash_lines = [
            "#!/usr/bin/env bash",
            "set -e",
            "",
            f"# Generated by MARL Experiment Agent at {now_str()}",
            f"# Total runs: {len(specs)}",
            "",
        ]
        for spec in specs:
            bash_lines.append(f"echo '[RUN] {spec.run_name}'")
            bash_lines.append(f"mkdir -p {quote_arg(str(spec.output_dir))}")
            bash_lines.append(self.command_to_shell(spec, cfg))
            bash_lines.append("")
        bash_path = outdir / "run_all.sh"
        write_text(bash_path, "\n".join(bash_lines))
        try:
            bash_path.chmod(0o755)
        except Exception:
            pass

        # A tmux helper: one session per run is too noisy, so create one window per algorithm.
        tmux_lines = [
            "#!/usr/bin/env bash",
            "set -e",
            f"SESSION={slugify(cfg.project_name)}",
            "tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION -n controller",
            "",
        ]
        for spec in specs:
            cmd = self.command_to_shell(spec, cfg).replace("'", "'\\''")
            tmux_lines.append(f"tmux new-window -t $SESSION -n {quote_arg(spec.run_name[:20])} '{cmd}'")
        tmux_lines.append("tmux attach -t $SESSION")
        tmux_path = outdir / "run_all_tmux.sh"
        write_text(tmux_path, "\n".join(tmux_lines))
        try:
            tmux_path.chmod(0o755)
        except Exception:
            pass

        # Windows cmd version. CUDA_VISIBLE_DEVICES syntax differs.
        bat_lines = [
            "@echo off",
            f"REM Generated by MARL Experiment Agent at {now_str()}",
            f"REM Total runs: {len(specs)}",
            "",
        ]
        for spec in specs:
            cmd = " ".join(str(x) for x in spec.command)
            bat_lines.append(f"echo [RUN] {spec.run_name}")
            bat_lines.append(f"mkdir {str(spec.output_dir)} 2>nul")
            bat_lines.append(f"set CUDA_VISIBLE_DEVICES={spec.gpu_id}")
            bat_lines.append(cmd)
            bat_lines.append("")
        bat_path = outdir / "run_all_windows.bat"
        write_text(bat_path, "\r\n".join(bat_lines))

        self.info(f"Scripts generated under {outdir}")
        return {"manifest": manifest_path, "bash": bash_path, "tmux": tmux_path, "windows_bat": bat_path}


# -----------------------------------------------------------------------------
# Log Analysis Agent
# -----------------------------------------------------------------------------


@dataclass
class ScalarSeries:
    metric: str
    steps: List[float]
    values: List[float]
    source_file: str

    def last_value(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    def tail_mean(self, n: int) -> Optional[float]:
        return mean(self.values[-n:]) if self.values else None

    def best_value(self, higher_is_better: bool = True) -> Optional[float]:
        if not self.values:
            return None
        return max(self.values) if higher_is_better else min(self.values)


@dataclass
class RunResult:
    algorithm: str
    scenario: str
    seed: Optional[int]
    run_name: str
    source_dir: str
    metrics: Dict[str, ScalarSeries] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def get_tail_mean(self, metric: str, n: int) -> Optional[float]:
        s = self.metrics.get(metric)
        return s.tail_mean(n) if s else None

    def get_last(self, metric: str) -> Optional[float]:
        s = self.metrics.get(metric)
        return s.last_value() if s else None


@dataclass
class AggregateResult:
    algorithm: str
    scenario: str
    metric: str
    values: List[float]
    mean: Optional[float]
    std: Optional[float]
    count: int


class LogAnalysisAgent(BaseAgent):
    name = "LogAnalysisAgent"

    def analyze(self, cfg: ExperimentConfig, logdir: Path) -> Tuple[List[RunResult], List[AggregateResult]]:
        run_dirs = self._discover_run_dirs(logdir)
        if not run_dirs:
            self.warn(f"No run directories found under {logdir}. Analysis will be empty.")
            return [], []

        results: List[RunResult] = []
        for rd in run_dirs:
            rr = self._analyze_run_dir(cfg, rd)
            if rr is not None:
                results.append(rr)

        aggs = self._aggregate(cfg, results)
        self.info(f"Analyzed {len(results)} runs from {logdir}.")
        return results, aggs

    def _discover_run_dirs(self, logdir: Path) -> List[Path]:
        if not logdir.exists():
            return []
        candidates: List[Path] = []
        # A run dir is any directory containing CSV files or TensorBoard event files.
        for root, dirs, files in os.walk(logdir):
            root_path = Path(root)
            if any(f.endswith(".csv") for f in files) or any(f.startswith("events.out.tfevents") for f in files):
                candidates.append(root_path)
        # Avoid nested duplicates: if parent is already included and child only contains copied logs, keep both only if distinct.
        return sorted(set(candidates))

    def _infer_metadata(self, cfg: ExperimentConfig, run_dir: Path) -> Tuple[str, str, Optional[int]]:
        name = run_dir.name
        algo = "UNKNOWN"
        scenario = "UNKNOWN"
        seed: Optional[int] = None

        upper_name = name.upper()
        for a in cfg.algorithms:
            if a.upper() in upper_name:
                algo = a
                break
        for s in cfg.scenarios:
            if s.name.lower() in name.lower():
                scenario = s.name
                break
        seed_match = re.search(r"seed[_\-]?(\d+)", name, flags=re.IGNORECASE)
        if seed_match:
            seed = int(seed_match.group(1))
        return algo, scenario, seed

    def _metric_aliases(self, cfg: ExperimentConfig) -> Dict[str, List[str]]:
        aliases = {k: list(v) for k, v in cfg.metric_aliases.items()}
        for k in list(aliases.keys()):
            if k not in aliases[k]:
                aliases[k].append(k)
        if cfg.primary_metric not in aliases:
            aliases[cfg.primary_metric] = [cfg.primary_metric]
        return aliases

    def _canonical_metric(self, cfg: ExperimentConfig, col: str) -> Optional[str]:
        aliases = self._metric_aliases(cfg)
        col_norm = col.strip().lower()
        for metric, names in aliases.items():
            if col_norm in [x.strip().lower() for x in names]:
                return metric
        return None

    def _analyze_run_dir(self, cfg: ExperimentConfig, run_dir: Path) -> Optional[RunResult]:
        algo, scenario, seed = self._infer_metadata(cfg, run_dir)
        rr = RunResult(
            algorithm=algo,
            scenario=scenario,
            seed=seed,
            run_name=run_dir.name,
            source_dir=str(run_dir),
        )

        csv_files = sorted(run_dir.glob("*.csv")) + sorted(run_dir.glob("**/*.csv"))
        csv_files = sorted(set(csv_files))
        for csv_file in csv_files:
            self._read_csv_scalars(cfg, csv_file, rr)

        # Optional TensorBoard support. Only used if installed.
        tb_files = sorted(run_dir.glob("events.out.tfevents*")) + sorted(run_dir.glob("**/events.out.tfevents*"))
        tb_files = sorted(set(tb_files))
        if tb_files:
            self._read_tensorboard_scalars(cfg, run_dir, rr)

        if not rr.metrics:
            rr.warnings.append("No scalar metrics found.")
            return rr

        if cfg.primary_metric not in rr.metrics:
            rr.warnings.append(f"Primary metric '{cfg.primary_metric}' not found.")
        return rr

    def _read_csv_scalars(self, cfg: ExperimentConfig, csv_file: Path, rr: RunResult) -> None:
        try:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return
                fieldnames = [x.strip() for x in reader.fieldnames]
                step_col = None
                for col in fieldnames:
                    if self._canonical_metric(cfg, col) == "step":
                        step_col = col
                        break
                canonical_cols: Dict[str, str] = {}
                for col in fieldnames:
                    cm = self._canonical_metric(cfg, col)
                    if cm and cm != "step":
                        canonical_cols[col] = cm

                rows = list(reader)
                if not rows:
                    return

                tmp: Dict[str, Tuple[List[float], List[float]]] = {}
                for idx, row in enumerate(rows):
                    step = safe_float(row.get(step_col)) if step_col else float(idx)
                    if step is None:
                        step = float(idx)
                    for col, metric in canonical_cols.items():
                        value = safe_float(row.get(col))
                        if value is None:
                            continue
                        if metric not in tmp:
                            tmp[metric] = ([], [])
                        tmp[metric][0].append(float(step))
                        tmp[metric][1].append(float(value))

                for metric, (steps, values) in tmp.items():
                    if not values:
                        continue
                    existing = rr.metrics.get(metric)
                    if existing is None or len(values) > len(existing.values):
                        rr.metrics[metric] = ScalarSeries(metric, steps, values, str(csv_file))
        except Exception as exc:
            rr.warnings.append(f"Failed to read CSV {csv_file}: {exc}")

    def _read_tensorboard_scalars(self, cfg: ExperimentConfig, run_dir: Path, rr: RunResult) -> None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
        except Exception:
            rr.warnings.append("TensorBoard event files found, but tensorboard package is not installed. Install: pip install tensorboard")
            return

        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for tag in tags:
                # Match exact tag suffix, e.g. eval/capture_rate -> capture_rate.
                cm = self._canonical_metric(cfg, tag)
                if cm is None:
                    cm = self._canonical_metric(cfg, tag.split("/")[-1])
                if cm is None or cm == "step":
                    continue
                events = ea.Scalars(tag)
                steps = [float(e.step) for e in events]
                values = [float(e.value) for e in events]
                if values:
                    existing = rr.metrics.get(cm)
                    if existing is None or len(values) > len(existing.values):
                        rr.metrics[cm] = ScalarSeries(cm, steps, values, f"tensorboard:{tag}")
        except Exception as exc:
            rr.warnings.append(f"Failed to read TensorBoard logs under {run_dir}: {exc}")

    def _aggregate(self, cfg: ExperimentConfig, results: List[RunResult]) -> List[AggregateResult]:
        groups: Dict[Tuple[str, str, str], List[float]] = {}
        metric_names = set([cfg.primary_metric])
        for r in results:
            metric_names.update(r.metrics.keys())
        for r in results:
            for metric in metric_names:
                value = r.get_tail_mean(metric, cfg.last_n_points)
                if value is None:
                    continue
                key = (r.algorithm, r.scenario, metric)
                groups.setdefault(key, []).append(value)
        aggs: List[AggregateResult] = []
        for (algo, scenario, metric), values in sorted(groups.items()):
            aggs.append(
                AggregateResult(
                    algorithm=algo,
                    scenario=scenario,
                    metric=metric,
                    values=values,
                    mean=mean(values),
                    std=std(values),
                    count=len(values),
                )
            )
        return aggs


# -----------------------------------------------------------------------------
# Review Agent
# -----------------------------------------------------------------------------


@dataclass
class ReviewFinding:
    level: str
    title: str
    detail: str


class ReviewAgent(BaseAgent):
    name = "ReviewAgent"

    def review(self, cfg: ExperimentConfig, results: List[RunResult], aggs: List[AggregateResult]) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []

        expected = len(cfg.algorithms) * len(cfg.scenarios) * len(cfg.seeds)
        if len(results) < expected:
            findings.append(
                ReviewFinding(
                    "WARN",
                    "Missing runs",
                    f"Expected {expected} runs from config, but only found {len(results)} analyzed run directories. "
                    "This may be normal if training is not finished, otherwise check log_dir and run_name conventions.",
                )
            )

        for r in results:
            for w in r.warnings:
                findings.append(ReviewFinding("WARN", f"Run warning: {r.run_name}", w))

        # Detect low seed count.
        group_counts: Dict[Tuple[str, str], int] = {}
        for r in results:
            group_counts[(r.algorithm, r.scenario)] = group_counts.get((r.algorithm, r.scenario), 0) + 1
        for algo in cfg.algorithms:
            for scenario in cfg.scenarios:
                count = group_counts.get((algo, scenario.name), 0)
                if count < cfg.min_expected_runs:
                    findings.append(
                        ReviewFinding(
                            "WARN",
                            "Insufficient repeated runs",
                            f"{algo} on {scenario.name} has {count} runs, below min_expected_runs={cfg.min_expected_runs}.",
                        )
                    )

        # Rank algorithms by primary metric for each scenario.
        by_scenario: Dict[str, List[AggregateResult]] = {}
        for a in aggs:
            if a.metric == cfg.primary_metric and a.mean is not None:
                by_scenario.setdefault(a.scenario, []).append(a)
        for scenario, rows in by_scenario.items():
            rows = sorted(rows, key=lambda x: x.mean if x.mean is not None else -1e18, reverse=cfg.higher_is_better)
            if rows:
                best = rows[0]
                findings.append(
                    ReviewFinding(
                        "INFO",
                        f"Best method on {scenario}",
                        f"{best.algorithm} has the best mean {cfg.primary_metric}={format_float(best.mean)} "
                        f"over {best.count} run(s). Check variance before making a strong claim.",
                    )
                )
                if len(rows) >= 2 and rows[0].mean is not None and rows[1].mean is not None:
                    gap = rows[0].mean - rows[1].mean if cfg.higher_is_better else rows[1].mean - rows[0].mean
                    if abs(gap) < 0.02:
                        findings.append(
                            ReviewFinding(
                                "WARN",
                                f"Small performance gap on {scenario}",
                                f"The top-2 gap is only {format_float(gap)}. Avoid claiming clear superiority without more seeds or significance tests.",
                            )
                        )

        if not aggs:
            findings.append(
                ReviewFinding(
                    "WARN",
                    "No aggregate metrics",
                    "No aggregate results were produced. Check whether your logs contain columns matching metric_aliases.",
                )
            )

        self.info(f"Generated {len(findings)} review findings.")
        return findings


# -----------------------------------------------------------------------------
# Report Agent
# -----------------------------------------------------------------------------


class ReportAgent(BaseAgent):
    name = "ReportAgent"

    def write_report(
        self,
        cfg: ExperimentConfig,
        outdir: Path,
        validation_messages: List[AgentMessage],
        script_paths: Optional[Dict[str, Path]],
        results: List[RunResult],
        aggs: List[AggregateResult],
        findings: List[ReviewFinding],
    ) -> Path:
        ensure_dir(outdir)
        report = self._build_report(cfg, validation_messages, script_paths, results, aggs, findings)
        report_path = outdir / "experiment_agent_report.md"
        write_text(report_path, report)

        # Also export aggregate table as CSV.
        agg_path = outdir / "aggregate_results.csv"
        self._write_aggregate_csv(agg_path, aggs)
        self.info(f"Report written to {report_path}")
        return report_path

    def _build_report(
        self,
        cfg: ExperimentConfig,
        validation_messages: List[AgentMessage],
        script_paths: Optional[Dict[str, Path]],
        results: List[RunResult],
        aggs: List[AggregateResult],
        findings: List[ReviewFinding],
    ) -> str:
        lines: List[str] = []
        lines.append(f"# MARL Experiment Agent Report")
        lines.append("")
        lines.append(f"- Project: `{cfg.project_name}`")
        lines.append(f"- Generated at: `{now_str()}`")
        lines.append(f"- Platform: `{platform.platform()}`")
        lines.append(f"- Primary metric: `{cfg.primary_metric}`")
        lines.append(f"- Last-N smoothing points: `{cfg.last_n_points}`")
        lines.append("")

        lines.append("## 1. Experiment Design")
        lines.append("")
        lines.append(f"Algorithms: {', '.join('`' + a + '`' for a in cfg.algorithms)}")
        lines.append("")
        lines.append("| Scenario | Difficulty | Key environment args |")
        lines.append("|---|---:|---|")
        for s in cfg.scenarios:
            args = ", ".join(f"{k}={v}" for k, v in s.env_args.items())
            lines.append(f"| `{s.name}` | {s.difficulty} | {args} |")
        lines.append("")
        lines.append(f"Seeds: `{cfg.seeds}`")
        lines.append("")

        lines.append("## 2. Generated Training Scripts")
        lines.append("")
        if script_paths:
            for k, p in script_paths.items():
                lines.append(f"- {k}: `{p}`")
        else:
            lines.append("No scripts were generated in this run.")
        lines.append("")

        lines.append("## 3. Validation and Agent Warnings")
        lines.append("")
        if validation_messages:
            lines.append("| Source | Level | Message |")
            lines.append("|---|---:|---|")
            for m in validation_messages:
                lines.append(f"| {m.source} | {m.level} | {m.message} |")
        else:
            lines.append("No validation messages.")
        lines.append("")

        lines.append("## 4. Aggregate Results")
        lines.append("")
        if aggs:
            lines.append("| Algorithm | Scenario | Metric | Mean | Std | Runs |")
            lines.append("|---|---|---|---:|---:|---:|")
            for a in sorted(aggs, key=lambda x: (x.scenario, x.metric, x.algorithm)):
                lines.append(
                    f"| {a.algorithm} | {a.scenario} | {a.metric} | "
                    f"{format_float(a.mean)} | {format_float(a.std)} | {a.count} |"
                )
        else:
            lines.append("No aggregate results available.")
        lines.append("")

        lines.append("## 5. Primary Metric Ranking")
        lines.append("")
        rank_lines = self._ranking_table(cfg, aggs)
        lines.extend(rank_lines)
        lines.append("")

        lines.append("## 6. Review Findings")
        lines.append("")
        if findings:
            lines.append("| Level | Finding | Detail |")
            lines.append("|---|---|---|")
            for f in findings:
                lines.append(f"| {f.level} | {f.title} | {f.detail} |")
        else:
            lines.append("No findings.")
        lines.append("")

        lines.append("## 7. Recommended Next Actions")
        lines.append("")
        lines.extend(self._recommendations(cfg, results, aggs, findings))
        lines.append("")

        lines.append("## 8. Reproducibility Notes")
        lines.append("")
        lines.append("- Keep the generated `run_manifest.json` with your final results.")
        lines.append("- Do not compare algorithms using only one seed unless this is explicitly a smoke test.")
        lines.append("- Report both mean and standard deviation over seeds.")
        lines.append("- For scalability experiments, separate in-distribution and zero-shot entity-count settings.")
        lines.append("- If using TensorBoard logs, install `tensorboard` so scalar event files can be parsed.")
        lines.append("")
        return "\n".join(lines)

    def _ranking_table(self, cfg: ExperimentConfig, aggs: List[AggregateResult]) -> List[str]:
        lines: List[str] = []
        by_scenario: Dict[str, List[AggregateResult]] = {}
        for a in aggs:
            if a.metric == cfg.primary_metric and a.mean is not None:
                by_scenario.setdefault(a.scenario, []).append(a)
        if not by_scenario:
            return [f"No ranking available for primary metric `{cfg.primary_metric}`."]
        for scenario, rows in sorted(by_scenario.items()):
            rows = sorted(rows, key=lambda x: x.mean if x.mean is not None else -1e18, reverse=cfg.higher_is_better)
            lines.append(f"### {scenario}")
            lines.append("")
            lines.append("| Rank | Algorithm | Mean | Std | Runs |")
            lines.append("|---:|---|---:|---:|---:|")
            for idx, r in enumerate(rows, start=1):
                lines.append(f"| {idx} | {r.algorithm} | {format_float(r.mean)} | {format_float(r.std)} | {r.count} |")
            lines.append("")
        return lines

    def _recommendations(
        self,
        cfg: ExperimentConfig,
        results: List[RunResult],
        aggs: List[AggregateResult],
        findings: List[ReviewFinding],
    ) -> List[str]:
        recs: List[str] = []
        warning_titles = {f.title for f in findings if f.level == "WARN"}
        if "Missing runs" in warning_titles:
            recs.append("- Complete missing training runs before drawing conclusions from the current report.")
        if any("Small performance gap" in f.title for f in findings):
            recs.append("- Increase the number of seeds or add statistical tests because the top algorithms are close.")
        if not aggs:
            recs.append("- First check log file column names and update `metric_aliases` in the config.")
        else:
            recs.append("- Use the aggregate table as the basis for paper tables, but manually verify abnormal runs before publication.")
        recs.append("- For zero-shot scalability, add scenarios with different `num_pursuers`, `num_evaders`, and `num_obstacles` while keeping the training scenario fixed.")
        recs.append("- Add task-specific metrics such as encirclement quality, time-to-capture, collision count, and boundary violation rate; reward alone is not enough.")
        return recs

    def _write_aggregate_csv(self, path: Path, aggs: List[AggregateResult]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "scenario", "metric", "mean", "std", "count", "values"])
            for a in aggs:
                writer.writerow([
                    a.algorithm,
                    a.scenario,
                    a.metric,
                    format_float(a.mean),
                    format_float(a.std),
                    a.count,
                    json.dumps(a.values),
                ])


# -----------------------------------------------------------------------------
# Optional Executor Agent
# -----------------------------------------------------------------------------


class ExecutorAgent(BaseAgent):
    name = "ExecutorAgent"

    def run_commands(self, cfg: ExperimentConfig, specs: List[RunSpec], max_runs: Optional[int] = None) -> None:
        """
        Execute generated commands directly.

        This is intentionally conservative. For real experiments, using run_all.sh/tmux is safer.
        """
        if cfg.training.dry_run:
            self.warn("training.dry_run=True, so commands will not be executed.")
            for s in specs[: max_runs or len(specs)]:
                print("DRY-RUN:", " ".join(quote_arg(x) for x in s.command))
            return

        selected = specs[: max_runs or len(specs)]
        for spec in selected:
            ensure_dir(spec.output_dir)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
            self.info(f"Executing {spec.run_name} on GPU {spec.gpu_id}")
            subprocess.run(spec.command, check=True, env=env)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


class ExperimentAgentSystem:
    def __init__(self) -> None:
        self.config_agent = ConfigAgent()
        self.script_agent = ScriptAgent()
        self.log_agent = LogAnalysisAgent()
        self.review_agent = ReviewAgent()
        self.report_agent = ReportAgent()
        self.executor_agent = ExecutorAgent()

    def init_config(self, out: Path) -> None:
        save_json(out, DEFAULT_CONFIG)
        print(f"Template config written to: {out}")

    def validate(self, config_path: Path) -> Tuple[ExperimentConfig, List[AgentMessage]]:
        cfg = self.config_agent.load(config_path)
        messages = self.config_agent.validate(cfg)
        self._print_messages(messages)
        if any(m.level == "ERROR" for m in messages):
            raise SystemExit("Config validation failed. Fix ERROR messages above.")
        return cfg, messages

    def plan(self, config_path: Path, outdir: Path) -> None:
        cfg, messages = self.validate(config_path)
        script_paths = self.script_agent.generate_scripts(cfg, outdir)
        for k, p in script_paths.items():
            print(f"{k}: {p}")
        self._print_messages(self.script_agent.messages)

    def analyze(self, config_path: Path, logdir: Path, outdir: Path) -> None:
        cfg, messages = self.validate(config_path)
        results, aggs = self.log_agent.analyze(cfg, logdir)
        findings = self.review_agent.review(cfg, results, aggs)
        report_path = self.report_agent.write_report(
            cfg=cfg,
            outdir=outdir,
            validation_messages=messages + self.log_agent.messages + self.review_agent.messages,
            script_paths=None,
            results=results,
            aggs=aggs,
            findings=findings,
        )
        print(f"Report: {report_path}")
        self._print_findings(findings)

    def all(self, config_path: Path, logdir: Path, outdir: Path) -> None:
        cfg, messages = self.validate(config_path)
        script_paths = self.script_agent.generate_scripts(cfg, outdir)
        results, aggs = self.log_agent.analyze(cfg, logdir)
        findings = self.review_agent.review(cfg, results, aggs)
        all_messages = (
            messages
            + self.script_agent.messages
            + self.log_agent.messages
            + self.review_agent.messages
            + self.report_agent.messages
        )
        report_path = self.report_agent.write_report(
            cfg=cfg,
            outdir=outdir,
            validation_messages=all_messages,
            script_paths=script_paths,
            results=results,
            aggs=aggs,
            findings=findings,
        )
        print(f"Report: {report_path}")
        print(f"Scripts: {outdir}")
        self._print_findings(findings)

    def run(self, config_path: Path, outdir: Path, max_runs: Optional[int] = None) -> None:
        cfg, _ = self.validate(config_path)
        specs = self.script_agent.build_run_specs(cfg, outdir)
        self.executor_agent.run_commands(cfg, specs, max_runs=max_runs)
        self._print_messages(self.executor_agent.messages)

    @staticmethod
    def _print_messages(messages: List[AgentMessage]) -> None:
        for m in messages:
            print(f"[{m.level}] {m.source}: {m.message}")

    @staticmethod
    def _print_findings(findings: List[ReviewFinding]) -> None:
        for f in findings:
            print(f"[{f.level}] {f.title}: {f.detail}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MARL Experiment Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a template experiment config")
    p_init.add_argument("--out", type=Path, default=Path("experiment_config.json"))

    p_val = sub.add_parser("validate", help="Validate config")
    p_val.add_argument("--config", type=Path, required=True)

    p_plan = sub.add_parser("plan", help="Generate run scripts and manifest")
    p_plan.add_argument("--config", type=Path, required=True)
    p_plan.add_argument("--outdir", type=Path, default=Path("agent_outputs"))

    p_analyze = sub.add_parser("analyze", help="Analyze logs and generate report")
    p_analyze.add_argument("--config", type=Path, required=True)
    p_analyze.add_argument("--logdir", type=Path, required=True)
    p_analyze.add_argument("--outdir", type=Path, default=Path("agent_outputs"))

    p_all = sub.add_parser("all", help="Generate scripts, analyze logs, and write report")
    p_all.add_argument("--config", type=Path, required=True)
    p_all.add_argument("--logdir", type=Path, required=True)
    p_all.add_argument("--outdir", type=Path, default=Path("agent_outputs"))

    p_run = sub.add_parser("run", help="Directly execute generated commands. Use carefully.")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--outdir", type=Path, default=Path("agent_outputs"))
    p_run.add_argument("--max-runs", type=int, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    system = ExperimentAgentSystem()

    if args.command == "init":
        system.init_config(args.out)
    elif args.command == "validate":
        system.validate(args.config)
    elif args.command == "plan":
        system.plan(args.config, args.outdir)
    elif args.command == "analyze":
        system.analyze(args.config, args.logdir, args.outdir)
    elif args.command == "all":
        system.all(args.config, args.logdir, args.outdir)
    elif args.command == "run":
        system.run(args.config, args.outdir, max_runs=args.max_runs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
