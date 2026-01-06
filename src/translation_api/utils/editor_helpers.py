"""Helper functions for editor comment endpoints."""

import re as _re
from typing import List


def extract_mentions(messages: list[dict], scope: str = "last", max_mentions: int = 5) -> list[str]:
    """Extract @mentions from messages."""
    pattern = _re.compile(r"@[A-Za-z0-9_\-]+")
    contents: list[str] = []
    if scope == "thread":
        contents = [m.content for m in messages if hasattr(m, "content")]
    else:
        if messages:
            contents = [messages[-1].content]
    seen = set()
    mentions: list[str] = []
    for c in contents:
        for h in pattern.findall(c or ""):
            if h not in seen:
                mentions.append(h)
                seen.add(h)
                if len(mentions) >= max_mentions:
                    return mentions
    return mentions


def enumerate_references(refs: list[dict]) -> tuple[str, list[str]]:
    """Enumerate references and return formatted section with IDs."""
    lines: list[str] = []
    ids: list[str] = []
    for i, r in enumerate(refs, start=1):
        t = (getattr(r, "type", None) or "ref").strip().lower()
        slug = _re.sub(r"[^a-z0-9\-]", "-", t) or "ref"
        rid = f"ref-{slug}-{i}"
        ids.append(rid)
        content = getattr(r, "content", "")
        lines.append(f"- [{rid}] (type={t}) {content}")
    section = "\n".join(lines) if lines else "None."
    return section, ids


def build_thread(messages: list[dict]) -> str:
    """Build thread text from messages."""
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_mentions_section(mentions: list[str]) -> str:
    """Build mentions section text."""
    if not mentions:
        return "None."
    return "\n".join([f"- {h}" for h in mentions])


def build_editor_prompt(thread_text: str, refs_section: str, mentions_section: str) -> str:
    """Build the editor comment prompt."""
    return (
        "You are a Tibetan Buddhist translation reviewer. Produce a concise, actionable commentary grounded ONLY in the provided references.\n\n"
        "TRIGGER\n"
        "- Proceed ONLY if the last message contains \"@Comment\".\n"
        "- If absent, return exactly:\n"
        '{"skipped": true, "reason": "No @Comment trigger"}\n'
        "and stop.\n\n"
        "THREAD (most recent last)\n"
        f"{thread_text}\n\n"
        "REFERENCES (use these IDs for citations)\n"
        f"{refs_section}\n\n"
        "MENTIONS\n"
        f"{mentions_section}\n\n"
        "TASK\n"
        "- Evaluate terminology, doctrinal accuracy, register, grammar, and consistency.\n"
        "- Make minimal, high-impact suggestions.\n"
        "- Every sentence MUST be supported by at least one reference and MUST end with bracketed citations using the exact IDs above (e.g., \"... [ref-commentary-1]\" or \"... [ref-scan-2;ref-lexicon-3]\").\n"
        "- If evidence is insufficient, do not make the claim. If critical context is missing, end with a short sentence requesting the needed references and cite [ref-needed].\n"
        "- If MENTIONS is non-empty, begin the comment with all handles in order, space-separated (e.g., \"@Kun @Tenzin \"), using the handles exactly.\n\n"
        "OUTPUT (JSON ONLY; single object)\n"
        "{\n"
        "  \"comment_text\": \"The full commentary with inline bracketed citations at the end of each sentence.\",\n"
        "  \"citations_used\": [\"ref-...\"]\n"
        "}\n\n"
        "RULES\n"
        "- Only output the JSON object above; no extra fields.\n"
        "- citations_used must be the unique set of IDs actually cited in comment_text.\n"
        "- Do not invent references or handles.\n"
    )


def build_editor_stream_prompt(thread_text: str, refs_section: str, mentions_section: str) -> str:
    """Build the editor comment streaming prompt."""
    return (
        "You are a Tibetan Buddhist translation reviewer. Produce a concise, actionable commentary grounded ONLY in the provided references.\n\n"
        "TRIGGER\n- Proceed ONLY if the last message contains \"@Comment\". If absent, output exactly: SKIP and stop.\n\n"
        "THREAD (most recent last)\n" + thread_text + "\n\n"
        "REFERENCES (use these IDs for citations)\n" + refs_section + "\n\n"
        "MENTIONS\n" + mentions_section + "\n\n"
        "TASK\n"
        "- Begin the comment with all handles in MENTIONS (space-separated) if any.\n"
        "- Make minimal, high-impact suggestions only.\n"
        "- Every sentence MUST end with bracketed citations using the exact IDs from REFERENCES, e.g., [ref-...;ref-...].\n"
        "- Do not add any preface or headers; output ONLY the final comment text.\n"
    )

