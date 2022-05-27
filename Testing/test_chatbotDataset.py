import unittest
from Scripts.chat_bot_dataset import ChatbotDataset

class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ChatbotDataset()
    
    def test_clean_text(self):
        text = 'What a GREAT day'
        expected = ['what', 'a', 'great', 'day']
        actual = self.dataset.clean_text(text)
        self.assertEqual(expected, actual)

    def test_bag(self):
        expected_words_in_bag = ['pasta', 'hello', 'relax']
        for word in expected_words_in_bag:
            self.assertIn(word, self.dataset.bag)


unittest.main(argv=[''], verbosity=2, exit=False)