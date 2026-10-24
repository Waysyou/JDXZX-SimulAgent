import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import v4_server as server
from src.app_config import RuntimeSettings


class V4ServerControlTests(unittest.TestCase):
    def tearDown(self):
        with server._sessions_lock:
            server._sessions.clear()

    def test_close_record_releases_resources_only_once(self):
        record = server.SessionRecord(state=app._new_live_state(), mode="microphone")
        with patch.object(
            server.pipeline, "release_live_session_resources"
        ) as release:
            server._close_record(record, remove_cache=True)
            server._close_record(record, remove_cache=True)

        release.assert_called_once_with(record.state, remove_cache=True)
        self.assertTrue(record.closed)
        self.assertTrue(record.resources_released)

    def test_register_session_evicts_oldest_record_at_configured_limit(self):
        first = server.SessionRecord(state=app._new_live_state(), mode="microphone")
        second = server.SessionRecord(state=app._new_live_state(), mode="microphone")
        settings = RuntimeSettings(project_root=server.PROJECT_ROOT, max_sessions=1)
        with (
            patch.object(server, "SETTINGS", settings),
            patch.object(server, "_close_record") as close_record,
        ):
            server._register_session(first)
            server._register_session(second)

        close_record.assert_called_once_with(first, remove_cache=True)
        self.assertNotIn(first.state["session_id"], server._sessions)
        self.assertIn(second.state["session_id"], server._sessions)

    def test_completed_file_worker_removes_upload_scratch_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = RuntimeSettings(project_root=Path(directory))
            upload_dir = settings.runtime_root / "v4_uploads" / "upload_1"
            upload_dir.mkdir(parents=True)
            (upload_dir / "sample.wav").write_bytes(b"wav")
            record = server.SessionRecord(
                state={"session_id": "file_session", "last_error": ""},
                mode="file",
                temp_paths={upload_dir},
            )
            with patch.object(server, "SETTINGS", settings):
                server._file_worker(record, iter(()))

            self.assertEqual(record.file_status, "DONE")
            self.assertFalse(upload_dir.exists())
            self.assertEqual(record.temp_paths, set())

    def test_completed_microphone_session_still_drains_pending_tts(self):
        state = app._new_live_state("microphone")
        state["finalization_status"] = "DONE"
        state["audio_queue"] = [{"segment_id": 2, "path": "tail.wav"}]
        record = server.SessionRecord(state=state, mode="microphone")

        with (
            patch.object(
                server.pipeline,
                "poll_microphone_finalization",
                return_value=(state,),
            ) as poll,
            patch.object(server, "_apply_live_outputs") as apply_outputs,
        ):
            server._poll_finalization_sync(record)

        poll.assert_called_once_with(state)
        apply_outputs.assert_called_once_with(record, (state,))


if __name__ == "__main__":
    unittest.main()
