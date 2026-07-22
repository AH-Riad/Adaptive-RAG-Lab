from src.loaders.text_loader import TextLoader

loader = TextLoader()

documents = loader.load("datasets/raw/sample.txt")
print(documents)
