"""Tests unitaires des garde-fous du moteur RAG TDR."""
import unittest

from src.tdr_rag.independent_engine import (
    _excerpt,
    _semantic_chunks,
    _verified_evidence,
)


class TDRRAGGuardTests(unittest.TestCase):
    def test_semantic_chunks_split_long_documents_with_overlap(self):
        text = ("Objectif spécifique et méthodologie de l'étude. " * 100)
        chunks = _semantic_chunks(text, size=300, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))
        self.assertTrue(all(chunks))

    def test_excerpt_finds_keywords_beyond_first_180_characters(self):
        text = (
            "Présentation générale. " + ("texte descriptif " * 30)
            + "\nObjectifs spécifiques : mesurer les résultats attendus."
            + "\nMéthodologie : enquête auprès des ménages."
        )
        excerpt = _excerpt(text, 500)
        self.assertIn("Objectifs spécifiques", excerpt)
        self.assertIn("Méthodologie", excerpt)

    def test_verified_evidence_rejects_invented_quotes(self):
        source = (
            "L'objectif général est d'évaluer les conditions de vie des ménages ruraux. "
            "La méthodologie repose sur une enquête auprès des ménages."
        )
        valid = "L'objectif général est d'évaluer les conditions de vie des ménages ruraux."
        invalid = "Le projet concerne la construction de routes nationales."
        result = _verified_evidence([valid, invalid], source)
        self.assertEqual(result, [valid])


if __name__ == "__main__":
    unittest.main()
