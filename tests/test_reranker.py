from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.retrieval.reranker import LocalCrossEncoderReranker, RerankerError


class FakeScores:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeCrossEncoder:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        **_kwargs: object,
    ) -> FakeScores:
        self.calls.append((pairs, batch_size))
        return FakeScores(self.values)


class LocalCrossEncoderRerankerTests(unittest.TestCase):
    def test_scores_pairs_with_lazy_local_model_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reranker = LocalCrossEncoderReranker(Path(temp_dir), batch_size=8)
            model = FakeCrossEncoder([0.2, 0.9])
            reranker._model = model

            scores = reranker.score("vacuum alarm", ["first", "second"])

            self.assertEqual(scores, [0.2, 0.9])
            self.assertEqual(model.calls[0][1], 8)
            self.assertEqual(
                model.calls[0][0],
                [("vacuum alarm", "first"), ("vacuum alarm", "second")],
            )

    def test_rejects_missing_local_model_before_runtime_load(self) -> None:
        reranker = LocalCrossEncoderReranker(Path("missing-reranker-model"))

        with self.assertRaisesRegex(RerankerError, "directory not found"):
            reranker.score("query", ["document"])

    def test_empty_documents_do_not_load_model(self) -> None:
        reranker = LocalCrossEncoderReranker(Path("missing-reranker-model"))

        self.assertEqual(reranker.score("query", []), [])
        self.assertFalse(reranker.is_loaded)


if __name__ == "__main__":
    unittest.main()
