from unittest import TestCase
from utility import MD_engine


class TestMD_engine(TestCase):

    def setup(self):
        self.engine = MD_engine()

    def test_check_sigma(self):
        self.setup()
        self.assertEqual(self.engine.check_sigma(), True)
