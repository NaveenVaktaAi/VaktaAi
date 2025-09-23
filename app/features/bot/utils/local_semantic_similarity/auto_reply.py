from pathlib import Path

from sentence_transformers.util import dot_score

from app.utils.transformers.models import msmarco_model


def _load_data(file: Path):
    return list({x for x in file.read_text(errors="ignore").splitlines() if x})


_DATA = _load_data(Path(__file__).parent.resolve() / "auto_reply.txt")
del _load_data

_sentences = msmarco_model.encode(_DATA, normalize_embeddings=True)


def can_auto_reply(sentence: str) -> bool:
    """Check if the sentence can automatically be replied to"""
    return (
        dot_score(
            msmarco_model.encode(sentence, normalize_embeddings=True), _sentences
        ).max()
        > 0.9
    )
