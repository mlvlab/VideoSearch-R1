import math
import os
import re

_PATTERN_THINK_TAG = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_PATTERN_START_TAG_ANY = re.compile(r"<start>\s*[\s\S]*?\s*</start>", re.IGNORECASE)
_PATTERN_END_TAG_ANY = re.compile(r"<end>\s*[\s\S]*?\s*</end>", re.IGNORECASE)
_PATTERN_START_TAG = re.compile(
    r"<start>\s*([-+]?\d+(?:\.\d+)?)\s*</start>", re.IGNORECASE
)
_PATTERN_END_TAG = re.compile(
    r"<end>\s*([-+]?\d+(?:\.\d+)?)\s*</end>", re.IGNORECASE
)


def _final_turn(turns):
    if not turns:
        return None
    for turn in reversed(turns):
        if not turn.get("shadow"):
            return turn
    return turns[-1]


def _extract_start_end_seconds(text):
    if not isinstance(text, str):
        return None
    start_hits = _PATTERN_START_TAG.findall(text)
    end_hits = _PATTERN_END_TAG.findall(text)
    if not start_hits or not end_hits:
        return None
    try:
        start = float(start_hits[-1])
        end = float(end_hits[-1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    return start, end


def _has_start_end_tags(text):
    if not isinstance(text, str):
        return False
    return bool(_PATTERN_START_TAG_ANY.search(text) and _PATTERN_END_TAG_ANY.search(text))


def _extract_gt_span(meta):
    raw = meta.get("gt_time", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        start = float(raw[0])
        end = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if end <= start:
        return None
    return start, end


def r_think_format(completions, **kwargs):
    """
    Reward 1.0 when <think>...</think> exists and content is non-empty; else 0.0.
    Uses the last think block so multi-turn concatenated completions are handled safely.
    """
    rewards = []
    for text in completions:
        if not isinstance(text, str):
            rewards.append(0.0)
            continue
        matches = _PATTERN_THINK_TAG.findall(text)
        if not matches:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if str(matches[-1]).strip() else 0.0)
    return rewards


def r_answer_binary(completions, solutions, problem_types, reward_meta=None, **kwargs):
    """Binary reward: +1 if answer matches label (matched iff retrieved_id==gt_video), else -1."""
    if reward_meta is None:
        return [0.0 for _ in completions]
    rewards = []
    for meta, solution in zip(reward_meta, solutions):
        turns = meta.get("turns", [])
        if not turns:
            rewards.append(0.0)
            continue
        final = _final_turn(turns)
        if not final:
            rewards.append(0.0)
            continue
        retrieved_id = str(final.get("retrieved_id", "") or "")
        gt_video = str(solution or "")
        if not gt_video:
            gt_video = str(meta.get("gt_video", "") or "")
        if not gt_video:
            gt_video = str(final.get("gt_video", "") or "")
        label = "matched" if retrieved_id and retrieved_id == gt_video else "not_matched"
        answer = str(final.get("answer", "") or "")
        rewards.append(1.0 if answer == label else -1.0)
    return rewards


def r_time_format(completions, solutions, problem_types, reward_meta=None, **kwargs):
    """
    Answer-conditioned timestamp format reward:
    - answer == not_matched:
        no timestamp tags => 1.0
        timestamp tags present => 0.0
    - answer == matched:
        timestamp tags present => 1.0
        no timestamp tags => 0.0
    """
    if reward_meta is None:
        return [0.0 for _ in completions]
    rewards = []
    for text, meta in zip(completions, reward_meta):
        turns = meta.get("turns", [])
        final = _final_turn(turns)
        if final is None:
            rewards.append(0.0)
            continue
        answer = str(final.get("answer", "") or "").strip().lower()
        has_tags = _has_start_end_tags(text)
        if answer == "not_matched":
            rewards.append(1.0 if not has_tags else 0.0)
        elif answer == "matched":
            rewards.append(1.0 if has_tags else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def r_time_iou(completions, solutions, problem_types, reward_meta=None, **kwargs):
    """
    Answer-conditioned temporal IoU reward:
    - answer == not_matched:
        no timestamp tags => 1.0
        timestamp tags present => 0.0
    - answer == matched:
        timestamp tags present => IoU(pred, gt)
        no timestamp tags => 0.0
    """
    if reward_meta is None:
        return [0.0 for _ in completions]
    rewards = []
    for text, meta in zip(completions, reward_meta):
        turns = meta.get("turns", [])
        final = _final_turn(turns)
        if final is None:
            rewards.append(0.0)
            continue
        # If answer correctness metadata exists, force zero IoU reward on wrong answers.
        # This keeps answer correctness and temporal IoU rewards separated.
        if ("correct_accept" in final) or ("correct_reject" in final):
            is_correct = bool(final.get("correct_accept", False)) or bool(
                final.get("correct_reject", False)
            )
            if not is_correct:
                rewards.append(0.0)
                continue
        answer = str(final.get("answer", "") or "").strip().lower()
        has_tags = _has_start_end_tags(text)
        if answer == "not_matched":
            rewards.append(0.5 if not has_tags else 0.0)
            continue
        if answer != "matched":
            rewards.append(0.0)
            continue
        if not has_tags:
            rewards.append(0.0)
            continue
        pred_span = _extract_start_end_seconds(text)
        gt_span = _extract_gt_span(meta)
        if pred_span is None or gt_span is None:
            rewards.append(0.0)
            continue
        p_start, p_end = pred_span
        g_start, g_end = gt_span
        if p_end <= p_start:
            rewards.append(0.0)
            continue
        inter = max(0.0, min(p_end, g_end) - max(p_start, g_start))
        union = max(p_end, g_end) - min(p_start, g_start)
        iou = inter / union if union > 0 else 0.0
        rewards.append(float(max(0.0, min(1.0, iou))))
    return rewards


def r_query_refine_quality(completions, solutions, problem_types, reward_meta=None, **kwargs):
    """
    Absolute query-refine quality reward injected by trainer:
    reward = [sim(q_final, v_gt) - temperature * logsumexp(sim(q_final, v_neg)/temperature)]
             * margin_reward_scale
    Notes:
      - Uses only q_final quality (no q_orig delta term).
      - Intended for query-refine-only optimization.
    """
    if reward_meta is None:
        return [0.0 for _ in completions]
    # If a sample did not emit <REFINE>, optionally apply a fixed fallback penalty/reward.
    # - 0.0  : ignore query-refine quality when no refine (no update from this term)
    # - -1.0 : penalize missing refine signal
    no_refine_value = float(os.environ.get("GRPO_NO_REFINE_QUALITY_PENALTY", "0.0") or 0.0)
    rewards = []
    for meta in reward_meta:
        if not bool(meta.get("improve_has_refine", False)):
            rewards.append(no_refine_value)
            continue
        rewards.append(float(meta.get("query_refine_quality", 0.0)))
    return rewards


def r_refine_presence(completions, solutions, problem_types, reward_meta=None, **kwargs):
    """Optional shaping reward for refine token generation."""
    if reward_meta is None:
        return [0.0 for _ in completions]
    rewards = []
    for meta in reward_meta:
        turns = meta.get("turns", [])
        final = _final_turn(turns)
        if final is not None and "has_refine_token" in final:
            rewards.append(1.0 if bool(final.get("has_refine_token", False)) else 0.0)
            continue
        rewards.append(1.0 if bool(meta.get("improve_has_refine", False)) else 0.0)
    return rewards


reward_funcs_registry = {
    "r_answer_binary": r_answer_binary,
    "r_think_format": r_think_format,
    "r_time_format": r_time_format,
    "r_time_iou": r_time_iou,
    "r_refine_presence": r_refine_presence,
    "r_query_refine_quality": r_query_refine_quality,
}
