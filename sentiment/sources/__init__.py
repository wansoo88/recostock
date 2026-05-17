"""Per-source fetchers. Each module exposes `fetch(...) -> list[dict]` where
each dict has at least `title` (str) and `published` (datetime | None)."""
