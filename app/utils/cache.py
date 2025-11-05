from functools import lru_cache


@lru_cache(maxsize=1024)
def get_cached_response(question: str):
# return None or string; ensure callers expect Optional[str]
    return None