"""Tests for DockerContainer lifecycle and container management."""

import unittest
from unittest.mock import MagicMock, patch

from docker_env.container import DockerContainer


class TestDockerContainer(unittest.TestCase):
    def test_container_name_sanitization(self):
        container = DockerContainer(
            image_name="swebench/test:latest",
            container_name="test/repo:name@123",
            auto_cleanup=False,
        )
        self.assertEqual(container.container_name, "test_repo_name_123")
        self.assertIsNone(container.container_id)

    def test_docker_available_boolean(self):
        # Should return a bool without raising exceptions
        available = DockerContainer.is_docker_available()
        self.assertIsInstance(available, bool)

    @patch("subprocess.run")
    def test_pull_image(self, mock_run):
        # image inspect returns 1 (not present), then docker pull returns 0
        def side_effect(cmd, *args, **kwargs):
            if cmd[1] == "image" and cmd[2] == "inspect":
                return MagicMock(returncode=1)
            if cmd[1] == "pull":
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        container = DockerContainer(image_name="test_img", auto_cleanup=False)
        self.assertTrue(container.pull_image())

    @patch("subprocess.run")
    def test_start_and_stop_lifecycle(self, mock_run):
        # 1. Mock docker info success
        # 2. Mock image inspect success
        # 3. Mock docker run returning container ID
        # 4. Mock docker rm returning 0
        def side_effect(cmd, *args, **kwargs):
            if cmd[1] == "info":
                return MagicMock(returncode=0)
            if cmd[1] == "image" and cmd[2] == "inspect":
                return MagicMock(returncode=0)
            if cmd[1] == "run":
                return MagicMock(returncode=0, stdout="mock_cid_12345\n")
            if cmd[1] == "rm":
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        container = DockerContainer(
            image_name="test_img",
            container_name="test_container",
            auto_cleanup=True,
        )

        cid = container.start()
        self.assertEqual(cid, "mock_cid_12345")
        self.assertEqual(container.container_id, "mock_cid_12345")
        self.assertIn("mock_cid_12345", DockerContainer._active_containers)

        container.stop()
        self.assertIsNone(container.container_id)
        self.assertNotIn("mock_cid_12345", DockerContainer._active_containers)


if __name__ == "__main__":
    unittest.main()
