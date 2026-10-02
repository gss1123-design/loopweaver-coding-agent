from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evals.runner import run_suite


class OfflineEvaluationTests(unittest.TestCase):
    def test_all_deterministic_cases_pass(self) -> None:
        results = asyncio.run(run_suite())
        self.assertEqual([result.case for result in results], ["direct_answer", "tool_roundtrip"])
        self.assertTrue(all(result.passed for result in results))

    def test_tool_roundtrip_records_tool_and_tokens(self) -> None:
        result = asyncio.run(run_suite("tool_roundtrip"))[0]
        self.assertTrue(result.passed)
        self.assertEqual(result.tool_names, ["get_value"])
        self.assertIsNotNone(result.trace)
        assert result.trace is not None
        self.assertEqual(result.trace["tool_calls"], 1)
        self.assertGreater(result.trace["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
