import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.long_term import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory = LongTermMemory(base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directories(self):
        self.assertTrue(os.path.exists(self.memory.knowledge_dir))
        self.assertTrue(os.path.exists(self.memory.milestones_dir))
        self.assertTrue(os.path.exists(self.memory.patterns_dir))

    def test_save_and_load_knowledge(self):
        data = {'key': 'value', 'nested': {'a': 1}}
        path = self.memory.save_knowledge('test_key', data)
        self.assertTrue(os.path.exists(path))

        loaded = self.memory.load_knowledge('test_key')
        self.assertEqual(loaded, data)

    def test_load_nonexistent_knowledge(self):
        result = self.memory.load_knowledge('nonexistent')
        self.assertIsNone(result)

    def test_list_knowledge_keys(self):
        self.memory.save_knowledge('key1', {'a': 1})
        self.memory.save_knowledge('key2', {'b': 2})
        keys = self.memory.list_knowledge_keys()
        self.assertIn('key1', keys)
        self.assertIn('key2', keys)

    def test_save_and_load_milestone(self):
        milestone = {'event': 'test', 'metrics': {'mae': 0.1}}
        path = self.memory.save_milestone(milestone)
        self.assertTrue(os.path.exists(path))

        milestones = self.memory.load_milestones()
        self.assertEqual(len(milestones), 1)
        self.assertEqual(milestones[0]['event'], 'test')

    def test_load_milestones_limit(self):
        for i in range(5):
            self.memory.save_milestone({'index': i})
        milestones = self.memory.load_milestones(limit=3)
        self.assertEqual(len(milestones), 3)

    def test_save_and_load_pattern(self):
        pattern = {'rule': 'if cpu > 0.9 then alert', 'confidence': 0.95}
        path = self.memory.save_pattern('cpu_alert', pattern)
        self.assertTrue(os.path.exists(path))

        loaded = self.memory.load_pattern('cpu_alert')
        self.assertEqual(loaded, pattern)

    def test_save_and_load_object(self):
        obj = {'complex': [1, 2, 3], 'nested': {'a': True}}
        path = self.memory.save_object('test_obj', obj)
        self.assertTrue(os.path.exists(path))

        loaded = self.memory.load_object('test_obj')
        self.assertEqual(loaded, obj)

    def test_delete(self):
        self.memory.save_knowledge('to_delete', {'a': 1})
        result = self.memory.delete('to_delete')
        self.assertTrue(result)
        self.assertIsNone(self.memory.load_knowledge('to_delete'))

    def test_delete_nonexistent(self):
        result = self.memory.delete('nonexistent')
        self.assertFalse(result)

    def test_storage_stats(self):
        self.memory.save_knowledge('k1', {'a': 1})
        self.memory.save_pattern('p1', {'b': 2})
        self.memory.save_milestone({'c': 3})

        stats = self.memory.get_storage_stats()
        self.assertIn('knowledge', stats)
        self.assertIn('milestones', stats)
        self.assertIn('patterns', stats)
        self.assertEqual(stats['knowledge']['file_count'], 1)
        self.assertEqual(stats['patterns']['file_count'], 1)
        self.assertEqual(stats['milestones']['file_count'], 1)

    def test_json_roundtrip(self):
        data = {'unicode': '中文测试', 'numbers': [1.5, 2.5, 3.5]}
        self.memory.save_knowledge('unicode_test', data)
        loaded = self.memory.load_knowledge('unicode_test')
        self.assertEqual(loaded, data)


if __name__ == '__main__':
    unittest.main()
