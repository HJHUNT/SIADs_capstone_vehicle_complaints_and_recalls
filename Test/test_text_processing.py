from nltk.corpus import stopwords

class TestNLTK:
    def test_stopwords(self):
        stop_words = set(stopwords.words("english"))
        print(stop_words)
        assert all(word == word.lower() for word in stop_words)