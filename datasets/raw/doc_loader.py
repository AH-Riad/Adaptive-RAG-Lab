from src.loaders.text_loader import TextLoader

loader = TextLoader()

documents = loader.load("datasets/sample.txt")

print(documents)