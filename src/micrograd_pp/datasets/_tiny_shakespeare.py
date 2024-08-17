import urllib.request


def load_tiny_shakespeare() -> str:
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "/tmp/tiny_shakespeare.txt",
    )
    with open("/tmp/tiny_shakespeare.txt", "r") as file:
        return file.read()
