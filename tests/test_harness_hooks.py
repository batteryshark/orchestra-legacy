from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from orchestra_cli import cli, db, harness_hooks


class HarnessHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".orchestra").mkdir()
        (self.root / ".orchestra" / "config.toml").write_text(
            '[settings]\ndefault_requester = "codex"\n', encoding="utf-8"
        )
        db.connect(self.root).close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, session: str, *, status: str = "done") -> int:
        con = db.connect(self.root)
        try:
            cur = con.execute(
                "INSERT INTO runs(agent,backend,requested_by,harness_session,workdir,"
                "status,started_at) VALUES('worker','codex','codex',?,?,?,?)",
                (session, str(self.root), status, db.now()),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()

    def _message(self, *, run_id: int, kind: str | None, body: str) -> int:
        con = db.connect(self.root)
        try:
            cur = con.execute(
                "INSERT INTO messages(sender,recipient,body,run_id,kind,created_at) "
                "VALUES('worker','codex',?,?,?,?)",
                (body, run_id, kind, db.now()),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()

    def test_claims_consultation_without_stealing_runner_interrupt(self) -> None:
        run_id = self._run("session-a")
        interrupt_id = self._message(
            run_id=run_id, kind="interrupt", body="[INTERRUPT] use v2"
        )
        consultation_id = self._message(
            run_id=run_id, kind="consult", body="CONSULT run 4: v1 or v2?"
        )

        prompt = harness_hooks.wait_for_attention(
            self.root, "session-a", poll_interval=0.01
        )

        self.assertIn("CONSULT run 4", prompt)
        con = db.connect(self.root)
        try:
            rows = {
                row["id"]: row["read_at"]
                for row in con.execute(
                    "SELECT id,read_at FROM messages WHERE id IN (?,?)",
                    (interrupt_id, consultation_id),
                )
            }
        finally:
            con.close()
        self.assertIsNone(rows[interrupt_id])
        self.assertIsNotNone(rows[consultation_id])

    def test_current_supervised_run_does_not_wait_on_itself(self) -> None:
        con = db.connect(self.root)
        try:
            cur = con.execute(
                "INSERT INTO runs(agent,backend,requested_by,harness_session,workdir,"
                "status,started_at) VALUES('codex','codex','codex','session-a',?,'running',?)",
                (str(self.root), db.now()),
            )
            con.commit()
            run_id = int(cur.lastrowid)
        finally:
            con.close()

        with mock.patch.dict(
            os.environ,
            {"ORCHESTRA_SELF": "codex", "ORCHESTRA_RUN_ID": str(run_id)},
            clear=False,
        ):
            self.assertIsNone(
                harness_hooks.wait_for_attention(
                    self.root, "session-a", poll_interval=0.01
                )
            )

    def test_different_session_cannot_claim_messages_or_wait_on_runs(self) -> None:
        run_id = self._run("session-a", status="running")
        message_id = self._message(
            run_id=run_id, kind="consult", body="session A only"
        )

        self.assertIsNone(
            harness_hooks.wait_for_attention(
                self.root, "session-b", poll_interval=0.01
            )
        )
        con = db.connect(self.root)
        try:
            self.assertIsNone(
                con.execute(
                    "SELECT read_at FROM messages WHERE id=?", (message_id,)
                ).fetchone()["read_at"]
            )
        finally:
            con.close()

        prompt = harness_hooks.wait_for_attention(
            self.root, "session-a", poll_interval=0.01
        )
        self.assertIn("session A only", prompt)

    def test_claude_binding_persists_session_environment(self) -> None:
        env_file = self.root / "claude-env"
        context = harness_hooks.bind_session("claude", "session-a", str(env_file))
        self.assertIn("--harness-session session-a", context)
        self.assertEqual(
            env_file.read_text(encoding="utf-8"),
            "export ORCHESTRA_HARNESS_SESSION=session-a\n",
        )

    def test_session_start_is_silent_outside_orchestra_project(self) -> None:
        args = Namespace(backend="codex", bind=True, session=None)
        output = io.StringIO()
        with (
            mock.patch.object(
                cli.paths, "find_root", side_effect=SystemExit("not an Orchestra project")
            ),
            mock.patch("sys.stdin", io.StringIO('{"session_id":"session-a"}')),
            contextlib.redirect_stdout(output),
        ):
            cli.cmd_hook(args)
        self.assertEqual(output.getvalue(), "")

    def test_backend_results_continue_only_when_attention_is_needed(self) -> None:
        prompt = "worker finished"
        self.assertEqual(harness_hooks.render_hook_result("opencode", prompt), prompt)
        self.assertEqual(harness_hooks.render_hook_result("opencode", None), "")
        codex = json.loads(harness_hooks.render_hook_result("codex", prompt))
        self.assertEqual(codex, {"decision": "block", "reason": prompt})
        self.assertEqual(json.loads(harness_hooks.render_hook_result("claude", None)), {})

    def test_install_merges_existing_hooks_and_preserves_other_settings(self) -> None:
        codex = self.root / "codex-home" / "hooks.json"
        codex.parent.mkdir()
        codex.write_text(
            json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "other"}]}]}}),
            encoding="utf-8",
        )
        claude = self.root / "claude-home" / "settings.json"
        claude.parent.mkdir()
        claude.write_text(json.dumps({"permissions": {"allow": ["Read"]}}), encoding="utf-8")

        with mock.patch.dict(
            os.environ,
            {
                "CODEX_HOME": str(codex.parent),
                "CLAUDE_CONFIG_DIR": str(claude.parent),
                "XDG_CONFIG_HOME": str(self.root / "xdg"),
            },
            clear=False,
        ):
            first = harness_hooks.install_global()
            second = harness_hooks.install_global()

        codex_data = json.loads(codex.read_text(encoding="utf-8"))
        claude_data = json.loads(claude.read_text(encoding="utf-8"))
        commands = [
            hook["command"]
            for group in codex_data["hooks"]["Stop"]
            for hook in group["hooks"]
        ]
        self.assertEqual(commands, ["other", "orchestra hook --backend codex"])
        start_commands = [
            hook["command"]
            for group in codex_data["hooks"]["SessionStart"]
            for hook in group["hooks"]
        ]
        self.assertEqual(start_commands, ["orchestra hook --backend codex --bind"])
        self.assertEqual(claude_data["permissions"], {"allow": ["Read"]})
        self.assertIn("installed", " ".join(first))
        self.assertTrue(all("already present" in status for status in second))
        self.assertIn(
            'event.type !== "session.idle"',
            (self.root / "xdg" / "opencode" / "plugins" / "orchestra.js").read_text(),
        )
        self.assertIn(
            '"shell.env"',
            (self.root / "xdg" / "opencode" / "plugins" / "orchestra.js").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
