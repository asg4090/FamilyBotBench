from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
DEFAULT_TEXT_FIELDS = ("doc", "text", "comment_body", "body", "content")
DEFAULT_USER_FIELDS = ("user_id", "author_id", "comment_author_id", "id")
DEFAULT_TOPIC_FIELDS = ("topic", "subreddit", "community", "category")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "you",
    "are",
    "was",
    "but",
    "not",
    "have",
    "they",
    "just",
    "like",
    "from",
    "what",
    "all",
    "about",
    "there",
}


@dataclass(frozen=True)
class HumanAnchor:
    user_id: str
    doc: str
    split: str = "train"
    topics: tuple[str, ...] = ()
    source_rows: int = 1


@dataclass(frozen=True)
class AnchorProfile:
    user_id: str
    split: str
    word_count: int
    comment_count: int
    mean_comment_words: float
    top_unigrams: list[str] = field(default_factory=list)
    top_bigrams: list[str] = field(default_factory=list)
    topic_summary: list[str] = field(default_factory=list)

    def to_safe_prompt_block(self) -> str:
        return (
            f"Anchor language profile for {self.user_id}:\n"
            f"- split: {self.split}\n"
            f"- word count: {self.word_count}\n"
            f"- mean comment words: {self.mean_comment_words:.1f}\n"
            f"- common single-word cues: {', '.join(self.top_unigrams)}\n"
            f"- common short phrases: {', '.join(self.top_bigrams)}\n"
            f"- topic cues: {', '.join(self.topic_summary)}\n"
            "- use this as aggregate style statistics only; do not copy source wording.\n"
        )


@dataclass(frozen=True)
class BotAnchorBinding:
    bot_user_id: str
    anchor_user_ids: list[str]
    anchor_split: str
    anchor_doc_word_count: int
    anchor_topic_summary: list[str]
    profiles: list[AnchorProfile]

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "bot_user_id": self.bot_user_id,
            "anchor_user_ids": "|".join(self.anchor_user_ids),
            "anchor_split": self.anchor_split,
            "anchor_doc_word_count": self.anchor_doc_word_count,
            "anchor_topic_summary": "|".join(self.anchor_topic_summary),
            "anchor_profile_json": json.dumps([asdict(profile) for profile in self.profiles], ensure_ascii=False),
        }


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(normalize_text(text))]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap_ratio(candidate: str, references: Iterable[str], *, n: int = 5) -> float:
    candidate_ngrams = ngrams(tokenize(candidate), n)
    if not candidate_ngrams:
        return 0.0
    reference_ngrams: set[tuple[str, ...]] = set()
    for ref in references:
        reference_ngrams.update(ngrams(tokenize(ref), n))
    if not reference_ngrams:
        return 0.0
    return len(candidate_ngrams & reference_ngrams) / len(candidate_ngrams)


def _first_present(row: Mapping[str, Any], fields: Sequence[str], default: str = "") -> str:
    for field_name in fields:
        if field_name in row and str(row[field_name]).strip():
            return str(row[field_name])
    return default


def _is_human_train_row(row: Mapping[str, Any], split: str, label_fields: Sequence[str]) -> bool:
    row_split = str(row.get("split", row.get("dataset_split", split))).lower()
    if row_split != split:
        return False
    for field_name in label_fields:
        if field_name not in row:
            continue
        label = str(row[field_name]).strip().lower()
        if label in {"human", "0", "false", "real"}:
            return True
        if label in {"bot", "1", "true", "synthetic"}:
            return False
        if "human" in label:
            return True
        if "bot" in label or "synthetic" in label or "generated" in label:
            return False
    return True


def build_human_anchor_pool(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_split: str = "train",
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    user_fields: Sequence[str] = DEFAULT_USER_FIELDS,
    topic_fields: Sequence[str] = DEFAULT_TOPIC_FIELDS,
    label_fields: Sequence[str] = ("label", "is_bot", "account_type", "source"),
    min_words: int = 20,
) -> list[HumanAnchor]:
    grouped_text: dict[str, list[str]] = defaultdict(list)
    grouped_topics: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    split = train_split.lower()

    for row in rows:
        if not _is_human_train_row(row, split, label_fields):
            continue
        user_id = _first_present(row, user_fields)
        text = normalize_text(_first_present(row, text_fields))
        if not user_id or not text:
            continue
        grouped_text[user_id].append(text)
        source_counts[user_id] += 1
        for field_name in topic_fields:
            topic = normalize_text(row.get(field_name, ""))
            if topic:
                grouped_topics[user_id][topic] += 1

    anchors: list[HumanAnchor] = []
    for user_id, docs in grouped_text.items():
        doc = " ".join(docs)
        if len(tokenize(doc)) < min_words:
            continue
        topics = tuple(topic for topic, _ in grouped_topics[user_id].most_common(8))
        anchors.append(HumanAnchor(user_id=user_id, doc=doc, split=split, topics=topics, source_rows=source_counts[user_id]))
    return sorted(anchors, key=lambda item: item.user_id)


def build_anchor_profile(anchor: HumanAnchor, *, max_terms: int = 12) -> AnchorProfile:
    tokens = tokenize(anchor.doc)
    unigram_counts = Counter(token for token in tokens if len(token) > 2 and token not in STOPWORDS)
    bigram_counts = Counter(" ".join(gram) for gram in ngrams(tokens, 2) if not all(token in STOPWORDS for token in gram))
    top_unigrams = [term for term, _ in unigram_counts.most_common(max_terms)]
    top_bigrams = [term for term, _ in bigram_counts.most_common(max_terms) if len(term.split()) == 2]
    return AnchorProfile(
        user_id=anchor.user_id,
        split=anchor.split,
        word_count=len(tokens),
        comment_count=anchor.source_rows,
        mean_comment_words=len(tokens) / max(1, anchor.source_rows),
        top_unigrams=top_unigrams[:max_terms],
        top_bigrams=top_bigrams[:max_terms],
        topic_summary=list(anchor.topics[:8]),
    )


def sample_anchor_bindings(
    bot_user_ids: Sequence[str],
    human_pool: Sequence[HumanAnchor],
    *,
    min_anchors: int = 2,
    max_anchors: int = 3,
    seed: int = 0,
) -> list[BotAnchorBinding]:
    if min_anchors < 1 or max_anchors < min_anchors:
        raise ValueError("anchor count must satisfy 1 <= min_anchors <= max_anchors")
    if len(human_pool) < min_anchors:
        raise ValueError(f"need at least {min_anchors} human anchors, got {len(human_pool)}")

    rng = random.Random(seed)
    bindings: list[BotAnchorBinding] = []
    for bot_user_id in bot_user_ids:
        anchor_count = rng.randint(min_anchors, min(max_anchors, len(human_pool)))
        anchors = rng.sample(list(human_pool), anchor_count)
        profiles = [build_anchor_profile(anchor) for anchor in anchors]
        topic_counts: Counter[str] = Counter()
        for profile in profiles:
            topic_counts.update(profile.topic_summary)
        bindings.append(
            BotAnchorBinding(
                bot_user_id=str(bot_user_id),
                anchor_user_ids=[anchor.user_id for anchor in anchors],
                anchor_split="train",
                anchor_doc_word_count=sum(profile.word_count for profile in profiles),
                anchor_topic_summary=[topic for topic, _ in topic_counts.most_common(12)],
                profiles=profiles,
            )
        )
    return bindings


def export_anchor_manifest(bindings: Sequence[BotAnchorBinding], path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bot_user_id",
        "anchor_user_ids",
        "anchor_split",
        "anchor_doc_word_count",
        "anchor_topic_summary",
        "anchor_profile_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for binding in bindings:
            writer.writerow(binding.to_manifest_row())


def build_anchor_profile_blocks(bindings: Sequence[BotAnchorBinding]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for binding in bindings:
        blocks[binding.bot_user_id] = "\n".join(profile.to_safe_prompt_block() for profile in binding.profiles)
    return blocks
