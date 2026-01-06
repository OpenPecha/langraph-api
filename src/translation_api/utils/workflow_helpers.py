"""Helper functions for workflow endpoints."""


def canonicalize_combo_key(combo_key: str) -> str:
    """Make combo_key order-independent by sorting tokens.

    Recognized tokens: source, ucca, gloss, sanskrit, commentariesK (K in {0,1,2,3,>3}).
    Any unknown tokens are preserved and sorted as well.
    """
    tokens = [t for t in combo_key.strip().split("+") if t]
    # Normalize 'commentaries' tokens to a count bucket if present
    normalized = []
    for t in tokens:
        if t.startswith("commentaries"):
            normalized.append(t)
        else:
            normalized.append(t)
    # Always include 'source' by default
    if "source" not in normalized:
        normalized.append("source")
    # Ensure unique and sorted
    normalized = sorted(set(normalized))
    return "+".join(normalized)


def derive_commentaries_token(comments: list[str] | None) -> str | None:
    """Derive commentaries token from comments list."""
    if comments is None:
        return None
    n = len(comments)
    if n > 3:
        return "commentaries>3"
    return f"commentaries{n}"

