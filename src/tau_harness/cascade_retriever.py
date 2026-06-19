"""Cascade retriever for workflow memory at agent session start.

The host-side wrapper that replaces ``WorkflowMemoryStore.retrieve``'s
lexical-Jaccard scoring with the three-stage cascade we measured to be
the best retriever in Phase 1.4–1.5:

  Stage 1 — Embedding cosine over all memories using a local
            sentence-transformers MiniLM model (no API call, ~10 ms CPU
            per query). The memory's ``embedding_text`` field is used
            when present; falls back to ``user_instruction``.

  Stage 2 — DELIBERATELY SKIPPED. The action-class filter was diagnosed
            as net-harmful in Phase 1.4 (dropped correct cross-class
            memories on ~4 covered tasks without adding safety). We may
            re-introduce a soft (fail-open-with-penalty) variant later;
            for now the cascade is two stages.

  Stage 3 — LLM rerank on the top-5 surviving candidates from stage 1.
            Default model: ``anthropic/claude-haiku-4.5`` via OpenRouter.
            The LLM can return NONE if no candidate fits — same option
            the audit script gave it. Cost ~$0.001 per session.

Why this lives in tau-agent and not kairos: the embedding-model
dependency (sentence-transformers, ~90 MB + PyTorch) is heavy. Hosts
that don't want it shouldn't have to install it. The cascade is a
host-side enhancement of a kairos primitive.

Public surface: :class:`CascadeMemoryStore` exposes the same
``retrieve(...)`` signature as :class:`kairos.semantic_recovery.memory.WorkflowMemoryStore`,
so it can drop in wherever a host configures memory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from kairos.models.semantic_recovery import MemoryRetrievalResult, WorkflowMemory
from kairos.semantic_recovery.memory import WorkflowMemoryStore

from tau_harness.openai_compat import build_client

_logger = logging.getLogger("tau_harness.cascade_retriever")

# Default rerank model. Kept here (not in kairos) because the cascade is a
# host-side concept; kairos itself is model-agnostic.
DEFAULT_RERANK_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_N_TO_RERANK = 5

_SYSTEM_PROMPT = """You are the final stage of a memory retrieval pipeline for an AI agent that handles tau-airline customer-service requests.

Earlier stages have already embedded the user request and scored candidate workflow memories by cosine similarity. You receive the user request and the top {N_CANDIDATES_PLACEHOLDER} most-relevant candidates. Your job: pick the SINGLE best match (top1) using the decision protocol below, with two alternates (top2, top3).

It is FAR better to return top1="NONE" than to inject a memory whose action class doesn't match the user's intent. Wrong-class injection misleads the agent (Phase 2.5 documented this on tau-bench tasks 13, 14, 27). When in doubt, NONE.

## Decision protocol (apply IN ORDER)

STEP 1 — Identify the user's INTENT ACTION CLASS from their stated request.

  Class          User's apparent goal                                  Cues in instruction
  ─────────────────────────────────────────────────────────────────────────────────────────
  book           Create a NEW reservation                              "fly from", "book a flight", "want to book"
  update         Modify an EXISTING reservation                        "change", "upgrade", "downgrade", "modify", "add bags"
  cancel         Cancel an existing reservation                        "cancel", "refund", "stop the trip"
  send_certificate  Compensation for delay/cancellation                "compensation", "voucher", "delayed flight"
  refuse-and-transfer  Policy-forbidden action OR compassionate-but-      basic-economy + change without insurance/24h/airline-
                       ineligible AND no candidate from same class as    cancel/business; "add insurance after booking";
                       gt fits                                           "remove bag"; "change passenger count";
                                                                         compassionate-reason for ineligible mutation
  read_only      Information only, no change                           "how many bags", "what is my", "tell me about"

  Multi-step compound (e.g., "cancel two and upgrade a third") usually classifies as the class of the FINAL/DOMINANT action.

STEP 2 — For each candidate, identify ITS action class from its `expected_tool_sequence` terminal:
  - book_* terminal → "book"
  - update_* terminal → "update"
  - cancel_reservation terminal → "cancel"
  - send_certificate terminal → "send_certificate"
  - transfer_to_human_agents terminal → "refuse-and-transfer"
  - No write/transfer terminal (only reads) → "read_only"

STEP 3 — FILTER candidates to those whose action class matches the user's intent class from STEP 1.

  - Multi-class candidates (e.g., cascade picked a "cancel + update" compound entry): match if the dominant terminal matches the user's intent class.
  - "refuse-and-transfer" candidates match the user's intent ONLY when their intent is in the refuse-and-transfer class. Do NOT default to refuse when the user has clear policy entitlement.

STEP 4 — If the FILTERED set is EMPTY (no candidate matches user's class), return top1="NONE".

  Do not soften this rule. A wrong-class memory entry actively misleads the agent into mutating-when-it-should-refuse or refusing-when-it-should-mutate. The agent has the wiki + the user's instruction; it can solve the task without a misleading memory.

STEP 5 — If the FILTERED set has 1+ candidates, pick the best by:
  (a) matching the user's specific CONSTRAINTS (insurance, cabin, payment method, threshold, etc.)
  (b) workflow-shape alignment (single-action vs compound, conditional vs unconditional)
  (c) negative_example aligns with what the user is trying to AVOID

  top2 and top3 fill from remaining filtered candidates, ranked by the same criteria.

## Critical policy entitlement check

When user's request involves modifying a basic-economy reservation, BEFORE classifying as "update", verify they have a policy carve-out:
  - travel insurance purchased on the reservation, AND
  - a health reason mentioned by the user (for cancel-with-insurance), OR
  - within the 24h booking window, OR
  - business cabin (basic-economy can't change but business can), OR
  - airline-cancelled flight

If NONE of these carve-outs apply and user wants modification → user's intent class is "refuse-and-transfer", not "update".

## Output schema

{
  "top1": "Mxx" or "NONE",
  "top2": "Mxx" or "NONE",
  "top3": "Mxx" or "NONE",
  "top1_confidence": "high" | "medium" | "low",
  "reason": "one sentence on why top1 was picked (or why NONE)"
}

Output ONLY the JSON object — no prose before or after, no code fences."""


def _short_code(idx: int) -> str:
    return f"M{idx + 1:02d}"


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


class CascadeMemoryStore:
    """Wraps a :class:`WorkflowMemoryStore` with the embedding+rerank cascade.

    Same ``retrieve(...)`` signature as the underlying store so it can be
    dropped into ``kairos.host.host(memory_store=...)`` transparently.

    Stage 1 (embeddings) is initialized once at construction. Stage 3
    (LLM rerank) is invoked on every ``retrieve`` call.

    The cascade ignores the ``min_semantic_score`` parameter because cosine
    scores aren't directly comparable to the lexical scorer's threshold,
    and the LLM rerank is responsible for the quality gate (it returns NONE
    when nothing fits, which produces zero hits).
    """

    def __init__(
        self,
        *,
        underlying: WorkflowMemoryStore,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        rerank_model: str | None = None,
        top_n_to_rerank: int = DEFAULT_TOP_N_TO_RERANK,
        rerank_provider: str | None = None,
        rerank_timeout_s: float = 30.0,
    ) -> None:
        # Lazy-import sentence-transformers so importing this module doesn't
        # cost the 200+ ms model-import overhead in callers that flip the
        # flag off at runtime.
        from sentence_transformers import SentenceTransformer

        self._underlying = underlying
        self._memories: list[WorkflowMemory] = list(underlying._memories)  # noqa: SLF001
        self._embed_model = SentenceTransformer(embed_model_name)
        self._top_n = top_n_to_rerank
        self._rerank_timeout_s = rerank_timeout_s

        # Provider routing: env var lets ops swap the rerank backend without
        # touching code (e.g. KAIROS_RERANKER_PROVIDER=azure with a gpt-4.1-mini
        # deployment). Default stays openrouter because the default rerank
        # model (Claude Haiku) only lives there.
        provider = rerank_provider or os.getenv("KAIROS_RERANKER_PROVIDER", "openrouter")
        self._rerank_provider = provider

        # Pick a model that matches the provider unless the caller forces one.
        if rerank_model is not None:
            self._rerank_model = rerank_model
        elif provider == "azure":
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_RERANKER")
            if not deployment:
                raise RuntimeError(
                    "rerank_provider=azure requires AZURE_OPENAI_DEPLOYMENT_RERANKER"
                )
            self._rerank_model = deployment
        else:
            self._rerank_model = DEFAULT_RERANK_MODEL

        self._openai = build_client(provider)

        # WorkflowMemory drops unknown fields (pydantic), so the JSON's
        # embedding_text and user_instruction don't survive ingestion. Re-read
        # the raw JSON file(s) and build a memory_id → embedding-text map by
        # matching trajectory_id (which DOES survive on memory.source_trajectory)
        # against the JSON's trajectory_id field.
        raw_text_by_traj_id: dict[str, str] = {}
        paths_text = os.getenv("KAIROS_WORKFLOW_MEMORY_PATHS", "")
        for path_str in paths_text.split(","):
            path_str = path_str.strip()
            if not path_str:
                continue
            try:
                entries = json.loads(Path(path_str).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                _logger.warning("cascade: failed to read %s: %s", path_str, exc)
                continue
            for entry in entries:
                tid = entry.get("trajectory_id")
                if not tid:
                    continue
                # Prefer embedding_text if curator supplied it; else
                # user_instruction (the curated playbook prose).
                text = entry.get("embedding_text") or entry.get("user_instruction") or ""
                if text:
                    raw_text_by_traj_id[tid] = text

        # Build the per-memory embedding text. Fall back through three
        # sources: explicit embedding_text > user_instruction > intent_signature.
        texts: list[str] = []
        n_using_explicit = 0
        for mem in self._memories:
            text = raw_text_by_traj_id.get(mem.source_trajectory, "")
            if text:
                n_using_explicit += 1
            else:
                text = mem.intent_signature or mem.title or ""
            texts.append(text)
        _logger.info(
            "cascade: %d/%d memories using explicit text from JSON; %d falling back to intent_signature",
            n_using_explicit,
            len(self._memories),
            len(self._memories) - n_using_explicit,
        )
        self._mem_embeddings = self._embed_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        _logger.info(
            "CascadeMemoryStore initialized — %d memories embedded with %s; rerank %s via %s",
            len(self._memories),
            embed_model_name,
            self._rerank_model,
            self._rerank_provider,
        )

    def _build_rerank_prompt(self, query: str, candidates: list[tuple[int, float]]) -> str:
        lines = [
            "USER REQUEST:",
            query,
            "",
            f"SURVIVING CANDIDATES (top {len(candidates)} by embedding cosine):",
        ]
        for idx, score in candidates:
            mem = self._memories[idx]
            lines.append("")
            lines.append(
                f"  {_short_code(idx)} [{mem.category}] {mem.title or '?'}  cosine={score:.3f}"
            )
            lines.append(
                f"        sequence: {' -> '.join(mem.expected_tool_sequence)}"
            )
            lines.append(
                f"        intent: {(mem.intent_signature or '')[:280]}"
            )
        lines.append("")
        lines.append("Pick the best, return JSON only.")
        return "\n".join(lines)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        toolset_hash: str | None = None,
        prompt_hash: str | None = None,
        min_semantic_score: float = -1.0,  # ignored by cascade
    ) -> list[MemoryRetrievalResult]:
        """Three-stage cascade retrieval.

        Returns up to ``top_k`` :class:`MemoryRetrievalResult` objects.
        Returns an empty list when stage 3 returns NONE — the
        :class:`SemanticRecoveryRuntime` and host plan-injection layer
        treat zero hits as "skip plan injection" (post commit c1c8006).
        """
        if not self._memories:
            return []

        # Stage 1: cosine over all memories
        q_emb = self._embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        cosine_scores = self._mem_embeddings @ q_emb
        ranked = sorted(enumerate(cosine_scores), key=lambda x: -x[1])

        # Stage 2: skipped (action-class filter was net-harmful)
        survivors = ranked[: self._top_n]

        # Stage 3: LLM rerank on the top-N
        prompt = self._build_rerank_prompt(query, survivors)
        try:
            resp = self._openai.chat.completions.create(
                model=self._rerank_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT.replace("{N_CANDIDATES_PLACEHOLDER}", str(self._top_n))},
                    {"role": "user", "content": prompt + "\n\nReturn ONLY the JSON object."},
                ],
                temperature=0.0,
                timeout=self._rerank_timeout_s,
            )
            raw = resp.choices[0].message.content or "{}"
        except Exception as exc:  # noqa: BLE001 - never crash on rerank failure
            _logger.warning("cascade rerank failed: %s; falling back to top cosine pick", exc)
            top_idx, top_score = survivors[0]
            return [self._build_result(top_idx, float(top_score))]

        parsed = _extract_json(raw)
        if not parsed:
            _logger.warning("cascade rerank returned unparseable text; falling back to top cosine")
            top_idx, top_score = survivors[0]
            return [self._build_result(top_idx, float(top_score))]

        # Build results from the LLM's top1/top2/top3 picks.
        # If NONE → return empty list (zero hits, no injection).
        candidate_codes = [_short_code(i) for i, _ in survivors]
        idx_by_code = {_short_code(i): (i, float(s)) for i, s in survivors}
        results: list[MemoryRetrievalResult] = []
        for slot in ("top1", "top2", "top3"):
            code = str(parsed.get(slot, "")).strip().upper()
            if code == "NONE" or not code:
                break  # stop at first NONE — LLM gave up
            if code in idx_by_code:
                idx, score = idx_by_code[code]
                results.append(self._build_result(idx, score))
            if len(results) >= top_k:
                break
        _logger.info(
            "cascade.retrieve query_head=%r candidates=%s → top1=%s confidence=%s",
            query[:80],
            candidate_codes,
            parsed.get("top1", ""),
            parsed.get("top1_confidence", ""),
        )
        return results

    def _build_result(self, idx: int, semantic_score: float) -> MemoryRetrievalResult:
        mem = self._memories[idx]
        return MemoryRetrievalResult(
            memory=mem,
            semantic_score=semantic_score,
            utility_score=mem.utility_score,
            matched_constraints=[],
            hash_compatible=True,
            source_trajectory=mem.source_trajectory,
        )
