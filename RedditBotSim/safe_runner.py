"""A controlled RedditBotSim runner with isolated outputs and dry-run support."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import ast
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REDDIT_BOT_SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REDDIT_BOT_SIM_DIR.parent
for rel in ["Action", "Environment", "Utils", "AgentDesicionCenter"]:
    sys.path.insert(0, str(REDDIT_BOT_SIM_DIR / rel))

from env import RedditEnv  # noqa: E402
from llm_client import build_llm_client, load_llm_settings  # noqa: E402
from reddit_agent import RedditAgent, create_reddit_user  # noqa: E402
from utils import load_config, set_logger  # noqa: E402


class DryRunClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": self})()

    @staticmethod
    def create(*args, **kwargs):
        message = type("Message", (), {
            "content": '["DRY_RUN_POST_ID", "DRY_RUN_TIME", "Dry-run generated comment."]'
        })()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _as_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with dst.open("ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def prepare_run_config(base_config_path: Path, run_dir: Path, max_agents: int | None) -> dict[str, Any]:
    config = load_config(str(base_config_path))
    run_data_dir = run_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = {
        "reddit_posts": run_data_dir / "Posts.csv",
        "reddit_comments1": run_data_dir / "Comment_1.csv",
        "reddit_comments2": run_data_dir / "Comment_2.csv",
        "reddit_users": run_data_dir / "User.csv",
    }
    for key, dst in copied_paths.items():
        _copy_csv(_resolve_path(config["paths"][key]), dst)
        config["paths"][key] = _as_project_relative(dst)

    if "Subreddit" in config["paths"]:
        subreddit_path = _resolve_path(config["paths"]["Subreddit"])
        if not subreddit_path.exists():
            fallback = REDDIT_BOT_SIM_DIR / "Data" / "RedditData" / "SubReddits.csv"
            config["paths"]["Subreddit"] = _as_project_relative(fallback)

    profile_path = _resolve_path(config["paths"]["agent_bots"])
    profile_rows = max(0, sum(1 for _ in profile_path.open(encoding="utf-8")) - 1)
    requested_agents = int(config.get("agents_num", profile_rows))
    if max_agents is not None:
        requested_agents = min(requested_agents, max_agents)
    config["agents_num"] = min(requested_agents, profile_rows)

    config.setdefault("log", {})
    config["log"]["log_file"] = str(run_dir / "log" / "redditbotsim.log")
    return config


def _parse_plan(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _fallback_comment1(recommend_posts: list) -> list[str]:
    selected_post = random.choice(recommend_posts)
    return [selected_post.submission_id, str(selected_post.time), "Fallback simulated first-level comment."]


def _fallback_comment2(recommend_posts: list) -> list[str] | None:
    posts_with_comments = [post for post in recommend_posts if post.comments1]
    if not posts_with_comments:
        return None
    selected_post = random.choice(posts_with_comments)
    selected_comment = random.choice(selected_post.comments1)
    return [
        selected_post.submission_id,
        selected_comment.comment1_id,
        str(selected_comment.comment1_time),
        "Fallback simulated second-level comment.",
    ]


def _json_default(value: Any) -> str:
    return str(value)


def write_trace(run_dir: Path, event: str, **payload: Any) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _extract_list(raw_text: str) -> Any:
    import re

    for match in re.findall(r"\[.*?\]", raw_text, flags=re.DOTALL):
        try:
            return ast.literal_eval(match)
        except Exception:
            continue
    return None


def _clean_identifier(value: Any) -> str:
    return str(value).strip().strip(".,;: ")


def call_comment1_llm(
    client: Any,
    model: str,
    temperature: float,
    reasoning_effort: str,
    prompt: str,
    recommend_posts_id_list: list[str],
    recommend_posts_time_list: list[str],
    api: str,
) -> tuple[str, list[str] | str | None]:
    raw_text = call_llm(client, model, temperature, reasoning_effort, prompt, api)
    if raw_text in ("Continue browsing", "End"):
        return raw_text, raw_text
    parsed = _extract_list(raw_text)
    if isinstance(parsed, list) and len(parsed) == 3:
        parsed[0] = _clean_identifier(parsed[0])
    if isinstance(parsed, list) and len(parsed) == 3 and parsed[0] in recommend_posts_id_list and parsed[1] in recommend_posts_time_list:
        return raw_text, parsed
    return raw_text, None


def call_comment2_llm(
    client: Any,
    model: str,
    temperature: float,
    reasoning_effort: str,
    prompt: str,
    recommend_posts_id_list: list[str],
    recommend_comments_id_list: list[str],
    recommend_comments_time_list: list[str],
    api: str,
) -> tuple[str, list[str] | str | None]:
    raw_text = call_llm(client, model, temperature, reasoning_effort, prompt, api)
    if raw_text in ("Continue browsing", "End"):
        return raw_text, raw_text
    parsed = _extract_list(raw_text)
    if isinstance(parsed, list) and len(parsed) == 4:
        parsed[0] = _clean_identifier(parsed[0])
        parsed[1] = _clean_identifier(parsed[1])
    if (
        isinstance(parsed, list)
        and len(parsed) == 4
        and parsed[0] in recommend_posts_id_list
        and parsed[1] in recommend_comments_id_list
        and parsed[2] in recommend_comments_time_list
    ):
        return raw_text, parsed
    return raw_text, None


def call_llm(client: Any, model: str, temperature: float, reasoning_effort: str, prompt: str, api: str) -> str:
    if hasattr(client, "generate_text"):
        return client.generate_text(prompt, api=api)
    if api == "responses":
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
        )
        return response.output_text
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def run_agents(config: dict[str, Any], client: Any, model: str, temperature: float, reasoning_effort: str, dry_run: bool, run_dir: Path, api: str) -> dict[str, int]:
    env = RedditEnv(config)
    env.reset()
    logger = set_logger(config, name="redditbotsim_safe_runner")
    logger.info("Safe runner started")
    logger.info("dry_run=%s model=%s reasoning=%s api=%s", dry_run, model, reasoning_effort, api)
    write_trace(run_dir, "run_started", dry_run=dry_run, model=model, reasoning_effort=reasoning_effort, api=api)

    df_agent_user = pd.read_csv(_resolve_path(config["paths"]["agent_bots"]), encoding="utf-8")
    total_user_nums = int(config["agents_num"])
    summary = {
        "agents_requested": total_user_nums,
        "agents_created": 0,
        "comment1_attempted": 0,
        "comment1_written": 0,
        "comment2_attempted": 0,
        "comment2_written": 0,
        "llm_calls_skipped_dry_run": 0,
        "llm_calls": 0,
        "llm_parse_failures": 0,
        "fallback_comments": 0,
    }

    for user_num in range(total_user_nums):
        row = df_agent_user.iloc[user_num]
        agent_columns = [
            "user_id",
            "user_name",
            "Description",
            "SubRedditList",
            "PostNum",
            "Comment1Num",
            "Comment2Num",
            "Preference",
        ]
        reddit_user, *_ = create_reddit_user(env, row[agent_columns])
        reddit_agent = RedditAgent(reddit_user, env)
        summary["agents_created"] += 1
        logger.info("Created agent %s (%s)", user_num, reddit_user.author_name)
        write_trace(
            run_dir,
            "agent_created",
            user_num=user_num,
            user_id=reddit_user.author_id,
            user_name=reddit_user.author_name,
            preference=reddit_user.character_setting,
            subreddit=reddit_user.subreddit,
            planned_comment1=row.get("Comment1Num"),
            planned_comment2=row.get("Comment2Num"),
        )

        profile = {
            "UserName": row["user_name"],
            "Gender": row["Gender"],
            "Age": row["Age"],
            "Region": row["Region"],
            "Education": row["EducationLevel"],
            "Ideology": row["Ideology"],
            "Description": row["Description"],
            "Event": row["Preference"],
        }

        for comment1_info in _parse_plan(row.get("Formulate_Comment1")):
            summary["comment1_attempted"] += 1
            planed_subreddit = comment1_info[2]
            planed_preference = comment1_info[3]
            planed_time = comment1_info[1]
            browse_start, browse_end = reddit_agent.set_read_post_time(planed_time, config)
            recommend_posts = reddit_agent.read(browse_start, browse_end)
            write_trace(
                run_dir,
                "browse_posts",
                action="comment1",
                planed_time=planed_time,
                browse_start=browse_start,
                browse_end=browse_end,
                recommended_count=len(recommend_posts) if isinstance(recommend_posts, list) else 0,
                recommended_post_ids=[post.submission_id for post in recommend_posts[:5]] if isinstance(recommend_posts, list) else [],
            )
            if not isinstance(recommend_posts, list) or not recommend_posts:
                continue

            if dry_run:
                new_comment = _fallback_comment1(recommend_posts)
                summary["llm_calls_skipped_dry_run"] += 1
            else:
                recommend_posts_id_list = [post.submission_id for post in recommend_posts]
                recommend_posts_time_list = [str(post.time) for post in recommend_posts]
                prompt = build_comment1_prompt(
                    profile, planed_preference, reddit_agent, recommend_posts, config
                )
                summary["llm_calls"] += 1
                write_trace(run_dir, "llm_prompt", action="comment1", prompt=prompt)
                raw_text, new_comment = call_comment1_llm(
                    client,
                    model,
                    temperature,
                    reasoning_effort,
                    prompt,
                    recommend_posts_id_list,
                    recommend_posts_time_list,
                    api,
                )
                write_trace(run_dir, "llm_response", action="comment1", raw_text=raw_text, parsed=new_comment)
                if new_comment in ("End", None):
                    summary["llm_parse_failures"] += 1
                    new_comment = _fallback_comment1(recommend_posts)
                    summary["fallback_comments"] += 1
                    write_trace(run_dir, "fallback_comment", action="comment1", fallback=new_comment)

            result = reddit_agent.comment1(new_comment, planed_subreddit)
            write_trace(run_dir, "comment_written", action="comment1", comment=new_comment, result=result)
            if result is not None:
                summary["comment1_written"] += 1

        for comment2_info in _parse_plan(row.get("Formulate_Comment2")):
            summary["comment2_attempted"] += 1
            planed_subreddit = comment2_info[2]
            planed_preference = comment2_info[3]
            planed_time = comment2_info[1]
            browse_start, browse_end = reddit_agent.set_read_post_time(planed_time, config)
            recommend_posts = reddit_agent.read(browse_start, browse_end)
            write_trace(
                run_dir,
                "browse_posts",
                action="comment2",
                planed_time=planed_time,
                browse_start=browse_start,
                browse_end=browse_end,
                recommended_count=len(recommend_posts) if isinstance(recommend_posts, list) else 0,
                recommended_post_ids=[post.submission_id for post in recommend_posts[:5]] if isinstance(recommend_posts, list) else [],
            )
            if not isinstance(recommend_posts, list) or not recommend_posts:
                continue

            if dry_run:
                new_comment = _fallback_comment2(recommend_posts)
                summary["llm_calls_skipped_dry_run"] += 1
            else:
                recommend_posts_id_list = [post.submission_id for post in recommend_posts]
                comments = [
                    comment
                    for post in recommend_posts
                    for comment in (post.comments1 or [])
                ]
                prompt = build_comment2_prompt(
                    profile, planed_preference, reddit_agent, recommend_posts, config
                )
                summary["llm_calls"] += 1
                write_trace(run_dir, "llm_prompt", action="comment2", prompt=prompt)
                raw_text, new_comment = call_comment2_llm(
                    client,
                    model,
                    temperature,
                    reasoning_effort,
                    prompt,
                    recommend_posts_id_list,
                    [comment.comment1_id for comment in comments],
                    [str(comment.comment1_time) for comment in comments],
                    api,
                )
                write_trace(run_dir, "llm_response", action="comment2", raw_text=raw_text, parsed=new_comment)
                if new_comment in ("End", None):
                    summary["llm_parse_failures"] += 1
                    new_comment = _fallback_comment2(recommend_posts)
                    summary["fallback_comments"] += 1
                    write_trace(run_dir, "fallback_comment", action="comment2", fallback=new_comment)

            result = reddit_agent.comment2(new_comment, planed_subreddit) if new_comment else None
            write_trace(run_dir, "comment_written", action="comment2", comment=new_comment, result=result)
            if result is not None:
                summary["comment2_written"] += 1

    logger.info("Safe runner finished: %s", summary)
    write_trace(run_dir, "run_finished", summary=summary)
    return summary


def build_comment1_prompt(profile, preference, reddit_agent, recommend_posts, config) -> str:
    return f"""You are {profile['UserName']}, gender {profile['Gender']}, age {profile['Age']}, from {profile['Region']}, education {profile['Education']}, ideology {profile['Ideology']}, description {profile['Description']}.
You must comment on one post about {preference}.
Return exactly one Python list in this format:
["PostID", "Post Time", "Comment Content"]

[Writing style]
Write like a normal Reddit user, not an assistant.
Use casual, concrete language. It is okay to be brief, skeptical, blunt, or a bit messy.
Prefer short sentences and specific reactions to the post.
Use contractions and ordinary first-person phrasing when it feels natural.
Do not imitate any provided user's exact style; there is no imitation target.
Do not summarize both sides like a neutral analyst.
Do not mention that you are an AI, a model, or generating content.
Avoid abstract essay words like "implications", "accountability", "conflict", "landscape", "ongoing", "ramifications", "critical", "strategy", "dialogue", "democracy", "military", "situation", "future", "witness", "stakes", "multifaceted", "underscores", and "highlights".
Avoid thesis-like wrap-ups such as "This highlights", "This underscores", or "complex dynamics".
Keep the comment roughly 1-3 sentences unless the thread clearly needs more.
Return exactly one Python list and nothing else.
Avoid square brackets inside the comment text.

[Posts]
{reddit_agent.get_recommend_posts(recommend_posts, 0)}
"""


def build_comment2_prompt(profile, preference, reddit_agent, recommend_posts, config) -> str:
    return f"""You are {profile['UserName']}, gender {profile['Gender']}, age {profile['Age']}, from {profile['Region']}, education {profile['Education']}, ideology {profile['Ideology']}, description {profile['Description']}.
You must reply to one first-level comment under a post about {preference}.
Return exactly one Python list in this format:
["Post ID", "First-Level Comment ID", "First-Level Comment Time", "Comment Content"]

[Writing style]
Reply to the selected first-level comment directly, as if continuing that thread.
Write like a normal Reddit user, not an assistant.
Use casual, concrete language. It is okay to be brief, skeptical, blunt, or a bit messy.
Prefer short sentences and specific reactions to the post/comment.
Use contractions and ordinary first-person phrasing when it feels natural.
Do not imitate any provided user's exact style; there is no imitation target.
Do not summarize both sides like a neutral analyst.
Do not mention that you are an AI, a model, or generating content.
Avoid abstract essay words like "implications", "accountability", "conflict", "landscape", "ongoing", "ramifications", "critical", "strategy", "dialogue", "democracy", "military", "situation", "future", "witness", "stakes", "multifaceted", "underscores", and "highlights".
Avoid thesis-like wrap-ups such as "This highlights", "This underscores", or "complex dynamics".
Keep the reply roughly 1-3 sentences unless the thread clearly needs more.
Return exactly one Python list and nothing else.
Avoid square brackets inside the comment text.

[Posts and comments]
{reddit_agent.get_recommend_posts(recommend_posts, 0)}
"""


def write_summary(run_dir: Path, summary: dict[str, int], config: dict[str, Any], dry_run: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    with (run_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# RedditBotSim run output\n\n")
        f.write(f"- dry_run: `{dry_run}`\n")
        f.write(f"- agents_num: `{config['agents_num']}`\n")
        f.write("- trace: `trace.jsonl`\n")
        f.write("- generated files are isolated under this run directory.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled RedditBotSim agent construction pass.")
    parser.add_argument("--config", default=str(REDDIT_BOT_SIM_DIR / "config.yml"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-agents", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and use deterministic fallback comments.")
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "outputs" / "redditbotsim_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    config = prepare_run_config(Path(args.config), run_dir, args.max_agents)

    if args.dry_run:
        client = DryRunClient()
        settings = load_llm_settings({"llm": {"api_key_env": "OPENAI_API_KEY"}}, provider_override=args.provider, model_override=args.model) if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY") else None
        model = settings.model if settings else "dry-run"
        temperature = settings.temperature if settings else 0.0
        reasoning_effort = settings.reasoning_effort if settings else "none"
    else:
        client, settings = build_llm_client(config, provider=args.provider, model=args.model)
        model = settings.model
        temperature = settings.temperature
        reasoning_effort = settings.reasoning_effort

    summary = run_agents(config, client, model, temperature, reasoning_effort, dry_run=args.dry_run, run_dir=run_dir, api=args.api)
    write_summary(run_dir, summary, config, args.dry_run)
    print(f"Run directory: {run_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
