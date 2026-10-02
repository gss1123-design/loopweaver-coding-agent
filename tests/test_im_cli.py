from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from im.cli import build_parser


class IMCliTests(unittest.TestCase):
    def test_tool_approval_flags_are_parsed(self) -> None:
        args = build_parser().parse_args(
            [
                "--tool-approval",
                "--approval-timeout-seconds",
                "12.5",
                "--llm-compaction",
            ]
        )
        self.assertTrue(args.tool_approval)
        self.assertEqual(args.approval_timeout_seconds, 12.5)
        self.assertTrue(args.llm_compaction)


if __name__ == "__main__":
    unittest.main()
