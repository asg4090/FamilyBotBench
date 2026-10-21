from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
AIISH_REPLACEMENTS = {
    "it is important to note": "",
    "it is worth noting": "",
    "in conclusion": "",
    "ultimately": "",
    "furthermore": "also",
    "moreover": "also",
    "delve into": "look at",
    "multifaceted": "messy",
    "nuanced": "complicated",
    "underscores": "shows",
    "highlights": "shows",
    "landscape": "scene",
    "implications": "fallout",
    "accountability": "answering for it",
    "ongoing": "still going",
}


@dataclass(frozen=True)
class LocalTfidfTerm:
    term: str
    candidate_tfidf: float
    anchor_centroid_tfidf: float
    lift: float
    anchor_present: bool


@dataclass(frozen=True)
class LocalTfidfDiagnostics:
    flagged_terms: list[LocalTfidfTerm]
    cosine_to_anchor_centroid: float
    anchor_overlap: dict[str, Any]
    candidate_token_count: int
    anchor_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "flagged_terms": [asdict(term) for term in self.flagged_terms],
            "cosine_to_anchor_centroid": self.cosine_to_anchor_centroid,
            "anchor_overlap": dict(self.anchor_overlap),
            "candidate_token_count": self.candidate_token_count,
            "anchor_token_count": self.anchor_token_count,
        }


@dataclass(frozen=True)
class RevisionTrace:
    iteration: int
    draft: str
    revised: str
    flagged_terms: list[dict[str, Any]]
    cosine_to_anchor_centroid: float
    anchor_overlap: dict[str, Any]
    rewrite_prompt: str
    changed: bool
    notes: list[str] = field(default_factory=list)
    detector_exposure: dict[str, Any] = field(default_factory=dict)
    llm_raw_response: str = ""
    llm_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"https?://\S+", " ", value)
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(normalize_text(text))]


def make_ngrams(tokens: Sequence[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def analyzer(text: str, *, ngram_range: tuple[int, int] = (1, 3)) -> list[str]:
    tokens = tokenize(text)
    terms: list[str] = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        terms.extend(make_ngrams(tokens, n))
    return terms


def compute_ngram_overlap(candidate: str, anchor_docs: Iterable[str], *, n: int = 5) -> dict[str, Any]:
    candidate_ngrams = set(make_ngrams(tokenize(candidate), n))
    anchor_ngrams: set[str] = set()
    for doc in anchor_docs:
        anchor_ngrams.update(make_ngrams(tokenize(doc), n))
    overlaps = sorted(candidate_ngrams & anchor_ngrams)
    return {
        "n": n,
        "candidate_ngram_count": len(candidate_ngrams),
        "anchor_ngram_count": len(anchor_ngrams),
        "overlap_count": len(overlaps),
        "overlap_ratio": len(overlaps) / max(1, len(candidate_ngrams)),
        "examples": overlaps[:10],
    }


def _tfidf_vectors(docs: Sequence[str], *, ngram_range: tuple[int, int]) -> tuple[list[dict[str, float]], dict[str, float]]:
    term_counts = [Counter(analyzer(doc, ngram_range=ngram_range)) for doc in docs]
    df: Counter[str] = Counter()
    for counts in term_counts:
        df.update(counts.keys())
    doc_count = len(docs)
    idf = {term: math.log((1 + doc_count) / (1 + freq)) + 1.0 for term, freq in df.items()}
    vectors: list[dict[str, float]] = []
    for counts in term_counts:
        total = sum(counts.values()) or 1
        vec = {term: (count / total) * idf[term] for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
        vectors.append({term: value / norm for term, value in vec.items()})
    return vectors, idf


def _centroid(vectors: Sequence[dict[str, float]]) -> dict[str, float]:
    total: Counter[str] = Counter()
    for vec in vectors:
        total.update(vec)
    denom = max(1, len(vectors))
    centroid = {term: value / denom for term, value in total.items()}
    norm = math.sqrt(sum(value * value for value in centroid.values())) or 1.0
    return {term: value / norm for term, value in centroid.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def fit_local_anchor_tfidf(
    anchor_docs: Sequence[str],
    candidate_doc: str,
    *,
    ngram_range: tuple[int, int] = (1, 3),
) -> dict[str, Any]:
    clean_anchors = [normalize_text(doc) for doc in anchor_docs if normalize_text(doc)]
    clean_candidate = normalize_text(candidate_doc)
    if not clean_anchors:
        raise ValueError("anchor_docs must contain at least one non-empty document")
    vectors, idf = _tfidf_vectors([*clean_anchors, clean_candidate], ngram_range=ngram_range)
    anchor_vectors = vectors[:-1]
    candidate_vector = vectors[-1]
    anchor_centroid = _centroid(anchor_vectors)
    return {
        "anchor_docs": clean_anchors,
        "candidate_doc": clean_candidate,
        "anchor_vectors": anchor_vectors,
        "candidate_vector": candidate_vector,
        "anchor_centroid": anchor_centroid,
        "idf": idf,
        "cosine_to_anchor_centroid": _cosine(candidate_vector, anchor_centroid),
    }


def diagnose_candidate_terms(
    local_model: dict[str, Any],
    *,
    top_k: int = 16,
    min_candidate_tfidf: float = 0.015,
    lift_threshold: float = 2.0,
) -> LocalTfidfDiagnostics:
    candidate_vector: dict[str, float] = local_model["candidate_vector"]
    anchor_centroid: dict[str, float] = local_model["anchor_centroid"]
    flagged: list[LocalTfidfTerm] = []
    for term, candidate_score in sorted(candidate_vector.items(), key=lambda item: item[1], reverse=True):
        anchor_score = anchor_centroid.get(term, 0.0)
        lift = candidate_score / max(anchor_score, 1e-9)
        if candidate_score < min_candidate_tfidf:
            continue
        if anchor_score > 0 and lift < lift_threshold:
            continue
        flagged.append(
            LocalTfidfTerm(
                term=term,
                candidate_tfidf=candidate_score,
                anchor_centroid_tfidf=anchor_score,
                lift=lift,
                anchor_present=anchor_score > 0,
            )
        )
        if len(flagged) >= top_k:
            break

    anchor_docs: list[str] = local_model["anchor_docs"]
    candidate_doc: str = local_model["candidate_doc"]
    return LocalTfidfDiagnostics(
        flagged_terms=flagged,
        cosine_to_anchor_centroid=float(local_model["cosine_to_anchor_centroid"]),
        anchor_overlap=compute_ngram_overlap(candidate_doc, anchor_docs, n=5),
        candidate_token_count=len(tokenize(candidate_doc)),
        anchor_token_count=sum(len(tokenize(doc)) for doc in anchor_docs),
    )


def render_rewrite_prompt(diagnostics: LocalTfidfDiagnostics) -> str:
    terms = [term.term for term in diagnostics.flagged_terms[:12]]
    return (
        "Revise the draft lightly so it reads like an ordinary Reddit comment. "
        "Keep the same stance and facts, avoid copying anchor text, and remove only obvious synthetic phrasing. "
        "Before accepting the revision, reread it once for fluency; it must remain grammatical, readable, and not broken by awkward deletions. "
        f"Reduce these locally over-prominent terms if doing so keeps the meaning: {terms}. "
        f"Reject any revision with 5-gram anchor overlap above 0.08. "
        "Return only the revised comment."
    )


def _anchor_excerpt(doc: str, *, max_words: int = 70) -> str:
    words = normalize_text(doc).split()
    return " ".join(words[:max_words])


def render_llm_revision_prompt(
    *,
    current_text: str,
    anchor_docs: Sequence[str],
    diagnostics: LocalTfidfDiagnostics,
    detector_exposure: Mapping[str, Any],
    iteration: int,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Prompt an LLM to revise after detector feedback.

    F3 is meant to be a revision loop, not a deletion loop: the bot sees why the
    current answer is exposed, then rewrites in a looser human-anchor voice.
    """
    metadata = dict(metadata or {})
    flagged_terms = [term.term for term in diagnostics.flagged_terms[:12]]
    detector_terms = [item.get("term", "") for item in detector_exposure.get("top_bot_terms", [])[:12]]
    anchor_blocks = []
    for idx, doc in enumerate(anchor_docs[:3], start=1):
        excerpt = _anchor_excerpt(doc)
        if excerpt:
            anchor_blocks.append(f"Anchor user {idx} style sample:\n{excerpt}")
    anchors = "\n\n".join(anchor_blocks) or "No anchor excerpt available."
    probability = detector_exposure.get("bot_probability", "unknown")
    predicted = detector_exposure.get("bot_prediction", "unknown")
    context = {
        "subreddit": metadata.get("subreddit", ""),
        "comment_level": metadata.get("level", ""),
        "parent_id": metadata.get("parent_id", ""),
        "link_id": metadata.get("link_id", ""),
    }
    return (
        "You are revising your own Reddit comment before it is logged in an offline social simulation.\n"
        "Your main goal is to move the language closer to the private human anchor voices. "
        "Your secondary goal is to avoid the old detector-exposed bot habits. "
        "Use both signals, but if they conflict, anchor fit and fluency matter more than simply lowering bot probability.\n\n"
        f"Iteration: {iteration + 1}\n"
        f"Local terms that do not fit the private human anchors: {flagged_terms}\n"
        f"Local cosine to anchor centroid: {diagnostics.cosine_to_anchor_centroid:.4f}\n"
        f"Secondary detector result: bot_probability={probability}, predicted_bot={predicted}\n"
        f"Secondary detector-exposed bot-leaning terms: {detector_terms}\n"
        f"Context metadata: {json.dumps(context, ensure_ascii=False)}\n\n"
        "Private human anchor voices. Loosely imitate their rhythm, concreteness, roughness, and level of effort. "
        "Do not copy any sentence or distinctive phrase from them.\n"
        f"{anchors}\n\n"
        "Rewrite goal:\n"
        "- Keep the same broad stance and do not invent new facts.\n"
        "- First priority: sound closer to the anchor users' ordinary rhythm, specificity, sentence length, and level of effort.\n"
        "- Second priority: avoid detector-exposed bot habits and old AI-like abstract wording.\n"
        "- Make it sound like a normal Reddit user responding in the moment.\n"
        "- Prefer concrete, situated wording over abstract policy-summary language.\n"
        "- It is okay to be casual, opinionated, skeptical, or a bit blunt if that fits the draft.\n"
        "- Before finalizing, reread the sentence once for fluency: the final content must be grammatical, readable, and not broken by awkward deletions.\n"
        "- Do not copy 5-word sequences from the anchor users.\n\n"
        f"Current draft:\n{normalize_text(current_text)}\n\n"
        "Return strict JSON only, exactly like this:\n"
        '{"content": "your revised Reddit comment"}'
    )


def diagnose_detector_exposure(
    vectorizer: Any,
    classifier: Any,
    candidate_doc: str,
    *,
    top_k: int = 14,
) -> dict[str, Any]:
    """Expose terms that push a TF-IDF LR detector toward the bot class."""
    doc = normalize_text(candidate_doc)
    x = vectorizer.transform([doc])
    probability = float(classifier.predict_proba(x)[0, 1])
    prediction = int(probability >= 0.5)
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_[0]
    row = x.tocsr()[0]
    top_terms: list[dict[str, Any]] = []
    for idx, value in zip(row.indices, row.data):
        contribution = float(value * coef[idx])
        if contribution <= 0:
            continue
        top_terms.append(
            {
                "term": str(feature_names[idx]),
                "tfidf": float(value),
                "coef": float(coef[idx]),
                "contribution": contribution,
            }
        )
    top_terms.sort(key=lambda item: item["contribution"], reverse=True)
    return {
        "bot_probability": probability,
        "bot_prediction": prediction,
        "top_bot_terms": top_terms[:top_k],
    }


def _parse_rewrite_content(raw_text: str) -> tuple[str, str | None]:
    text = str(raw_text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return normalize_text(obj.get("content", "")), None
    except Exception as exc:
        parse_error = f"json_parse_error: {exc}"
    else:
        parse_error = "json_parse_error: top-level value is not an object"
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return normalize_text(obj.get("content", "")), None
        except Exception as exc:
            return "", f"json_extract_error: {exc}"
    return "", parse_error


def rewrite_with_llm_feedback(
    anchor_docs: Sequence[str],
    candidate_doc: str,
    *,
    llm_client: Any,
    detector: tuple[Any, Any] | None = None,
    iteration: int = 0,
    api: str = "responses",
    settings_override: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ngram_range: tuple[int, int] = (1, 3),
    top_k: int = 16,
    overlap_limit: float = 0.08,
) -> tuple[str, RevisionTrace]:
    current = normalize_text(candidate_doc)
    local_model = fit_local_anchor_tfidf(anchor_docs, current, ngram_range=ngram_range)
    diagnostics = diagnose_candidate_terms(local_model, top_k=top_k)
    detector_exposure: dict[str, Any] = {}
    if detector is not None:
        vectorizer, classifier = detector
        detector_exposure = diagnose_detector_exposure(vectorizer, classifier, current)
    prompt = render_llm_revision_prompt(
        current_text=current,
        anchor_docs=anchor_docs,
        diagnostics=diagnostics,
        detector_exposure=detector_exposure,
        iteration=iteration,
        metadata=metadata,
    )
    raw_response = ""
    llm_error: str | None = None
    notes: list[str] = []
    try:
        if hasattr(llm_client, "generate_text"):
            raw_response = llm_client.generate_text(prompt, api=api, settings_override=settings_override)
        else:
            raw_response = llm_client.responses.create(input=prompt, model=(settings_override or {}).get("model")).output_text
    except Exception as exc:
        llm_error = str(exc)
        notes.append(f"llm call failed: {llm_error}")

    revised, parse_error = _parse_rewrite_content(raw_response)
    if parse_error is not None:
        llm_error = parse_error if llm_error is None else f"{llm_error}; {parse_error}"
        notes.append(parse_error)
        revised = current
    if not revised:
        notes.append("empty llm revision")
        revised = current

    overlap = compute_ngram_overlap(revised, anchor_docs, n=5)
    if overlap["overlap_ratio"] > overlap_limit:
        notes.append("rejected llm rewrite because anchor 5-gram overlap is too high")
        revised = current
        overlap = diagnostics.anchor_overlap
    quality_reason = quality_rejection_reason(current, revised)
    if quality_reason is not None:
        notes.append(f"rejected llm rewrite because {quality_reason}")
        revised = current
        overlap = diagnostics.anchor_overlap

    trace = RevisionTrace(
        iteration=iteration,
        draft=current,
        revised=revised,
        flagged_terms=[asdict(term) for term in diagnostics.flagged_terms],
        cosine_to_anchor_centroid=diagnostics.cosine_to_anchor_centroid,
        anchor_overlap=overlap,
        rewrite_prompt=prompt,
        changed=revised != current,
        notes=notes,
        detector_exposure=detector_exposure,
        llm_raw_response=raw_response,
        llm_error=llm_error,
    )
    return revised, trace


def run_llm_revision_loop(
    anchor_docs: Sequence[str],
    candidate_doc: str,
    *,
    llm_client: Any,
    detector: tuple[Any, Any] | None = None,
    max_iters: int = 2,
    trace_path: Path | str | None = None,
    api: str = "responses",
    settings_override: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = normalize_text(candidate_doc)
    traces: list[RevisionTrace] = []
    for iteration in range(max_iters):
        revised, trace = rewrite_with_llm_feedback(
            anchor_docs,
            current,
            llm_client=llm_client,
            detector=detector,
            iteration=iteration,
            api=api,
            settings_override=settings_override,
            metadata=metadata,
        )
        traces.append(trace)
        if trace_path is not None:
            append_revision_trace(trace_path, trace)
        current = revised
    final_exposure: dict[str, Any] = {}
    if detector is not None:
        vectorizer, classifier = detector
        final_exposure = diagnose_detector_exposure(vectorizer, classifier, current)
    return {
        "draft": normalize_text(candidate_doc),
        "final_text": current,
        "iterations": len(traces),
        "changed": current != normalize_text(candidate_doc),
        "revision_trace": [trace.to_dict() for trace in traces],
        "final_detector_exposure": final_exposure,
    }


def _replace_case_insensitive(text: str, phrase: str, replacement: str) -> str:
    return re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)


def light_rewrite_text(text: str, flagged_terms: Sequence[LocalTfidfTerm], *, max_term_edits: int = 6) -> tuple[str, list[str]]:
    revised = normalize_text(text)
    notes: list[str] = []
    revised = re.sub(r"\[[^\]]*\]", "", revised)
    revised = re.sub(r"\b(as an ai|i cannot|i can't browse|language model)\b", "", revised, flags=re.IGNORECASE)
    for phrase, replacement in AIISH_REPLACEMENTS.items():
        updated = _replace_case_insensitive(revised, phrase, replacement)
        if updated != revised:
            notes.append(f"replaced phrase: {phrase}")
            revised = updated

    edits = 0
    for term in flagged_terms:
        if edits >= max_term_edits:
            break
        if " " not in term.term:
            continue
        if term.anchor_present:
            continue
        if len(term.term.split()) > 3:
            continue
        pattern = re.compile(r"\b" + re.escape(term.term) + r"\b", flags=re.IGNORECASE)
        updated = pattern.sub("", revised, count=1)
        if updated != revised:
            notes.append(f"removed local outlier phrase: {term.term}")
            revised = updated
            edits += 1

    revised = re.sub(r"\s+([,.;!?])", r"\1", revised)
    revised = re.sub(r"\s{2,}", " ", revised).strip()
    return revised, notes


def quality_rejection_reason(original: str, revised: str) -> str | None:
    original_clean = normalize_text(original)
    revised_clean = normalize_text(revised)
    original_tokens = tokenize(original_clean)
    revised_tokens = tokenize(revised_clean)
    if original_clean and not revised_clean:
        return "empty revision"
    if len(original_tokens) >= 8 and len(revised_tokens) < max(5, int(len(original_tokens) * 0.6)):
        return "revision removed too much text"
    if len(original_clean) >= 60 and len(revised_clean) < max(30, int(len(original_clean) * 0.5)):
        return "revision is much shorter than draft"
    if len(revised_tokens) >= 4:
        unique_ratio = len(set(revised_tokens)) / max(1, len(revised_tokens))
        if unique_ratio < 0.45:
            return "revision has too many repeated tokens"
        bigrams = make_ngrams(revised_tokens, 2)
        if bigrams:
            counts = Counter(bigrams)
            repeated = sum(count - 1 for count in counts.values() if count > 1) / len(bigrams)
            if repeated > 0.25:
                return "revision has repeated phrase fragments"
    if re.search(r"\b([A-Za-z']+)\s+\1\b", revised_clean, flags=re.IGNORECASE):
        return "revision repeats adjacent words"
    if re.search(r"\b(at|like)\s+some\s+point\s+(are|is|was|were)\b", revised_clean, flags=re.IGNORECASE):
        return "revision has a broken clause after removing a phrase"
    if re.search(r"\b(point|thing|part|reason)\s+(are|is|was|were)\s+(dead|gone|killed|missing)\b", revised_clean, flags=re.IGNORECASE):
        return "revision has a broken copular clause"
    return None


def rewrite_with_diagnostics(
    anchor_docs: Sequence[str],
    candidate_doc: str,
    *,
    iteration: int = 0,
    ngram_range: tuple[int, int] = (1, 3),
    top_k: int = 16,
    overlap_limit: float = 0.08,
) -> tuple[str, RevisionTrace]:
    local_model = fit_local_anchor_tfidf(anchor_docs, candidate_doc, ngram_range=ngram_range)
    diagnostics = diagnose_candidate_terms(local_model, top_k=top_k)
    revised, notes = light_rewrite_text(candidate_doc, diagnostics.flagged_terms)
    overlap = compute_ngram_overlap(revised, anchor_docs, n=5)
    if overlap["overlap_ratio"] > overlap_limit:
        notes.append("rejected rewrite because anchor 5-gram overlap is too high")
        revised = normalize_text(candidate_doc)
        overlap = diagnostics.anchor_overlap
    quality_reason = quality_rejection_reason(candidate_doc, revised)
    if quality_reason is not None:
        notes.append(f"rejected rewrite because {quality_reason}")
        revised = normalize_text(candidate_doc)
        overlap = diagnostics.anchor_overlap
    trace = RevisionTrace(
        iteration=iteration,
        draft=normalize_text(candidate_doc),
        revised=revised,
        flagged_terms=[asdict(term) for term in diagnostics.flagged_terms],
        cosine_to_anchor_centroid=diagnostics.cosine_to_anchor_centroid,
        anchor_overlap=overlap,
        rewrite_prompt=render_rewrite_prompt(diagnostics),
        changed=revised != normalize_text(candidate_doc),
        notes=notes,
    )
    return revised, trace


def run_revision_loop(
    anchor_docs: Sequence[str],
    candidate_doc: str,
    *,
    max_iters: int = 3,
    min_similarity_gain: float = 0.005,
    trace_path: Path | str | None = None,
) -> dict[str, Any]:
    current = normalize_text(candidate_doc)
    traces: list[RevisionTrace] = []
    previous_similarity = -1.0
    for iteration in range(max_iters):
        revised, trace = rewrite_with_diagnostics(anchor_docs, current, iteration=iteration)
        traces.append(trace)
        if trace_path is not None:
            append_revision_trace(trace_path, trace)
        similarity = trace.cosine_to_anchor_centroid
        if not trace.changed:
            break
        if previous_similarity >= 0 and similarity - previous_similarity < min_similarity_gain:
            current = revised
            break
        previous_similarity = similarity
        current = revised
    return {
        "draft": normalize_text(candidate_doc),
        "final_text": current,
        "iterations": len(traces),
        "changed": current != normalize_text(candidate_doc),
        "revision_trace": [trace.to_dict() for trace in traces],
    }


def append_revision_trace(path: Path | str, trace: RevisionTrace) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
