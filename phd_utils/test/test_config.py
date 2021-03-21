import unittest

from phd_utils import global_config

class TestConfig(unittest.TestCase):
    
    def test_get_attribute(self):
        global_config.models.path
    
    def test_get_item(self):
        global_config.models['path']