import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ProcessRecoveryTests(unittest.TestCase):
    def test_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-m', 'learning_agent', 'alice', '--demo', '--data-dir', directory]
            first = subprocess.run(command, input='我叫小林。\n/note\n进程恢复测试\n审批后才能写入。\nexit\n',
                                   text=True, capture_output=True, timeout=45)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(list(Path(directory).rglob('*.md')), [])
            second = subprocess.run(command, input='我叫什么名字？\n/approve latest\nexit\n',
                                    text=True, capture_output=True, timeout=45)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn('你叫小林', second.stdout)
            files = list(Path(directory).rglob('*.md'))
            self.assertEqual(len(files), 1)
            self.assertIn('审批后才能写入', files[0].read_text(encoding='utf-8'))
