import os
from pathlib import Path
import urllib.request
import zipfile
import ssl
from beir.datasets.data_loader import GenericDataLoader
from src.core import Document

class BEIRDataset:
    def __init__(self, name: str = "fiqa", data_dir: str = "datasets/external"):
        self.name = name
        self.data_dir = data_dir
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    def load(self, split: str = "test"):
        out_dir = Path(self.data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{self.name}.zip"
        data_folder = out_dir / self.name
        
        if not data_folder.exists():
            print(f"Downloading {self.name} dataset (this may take a minute)...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.name}.zip"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            # Bypass Windows SSL certificate verification error
            context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(req, context=context) as response:
                with open(zip_path, 'wb') as out_file:
                    out_file.write(response.read())
                    
            print(f"Unzipping {self.name} dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(out_dir)
        
        # Load standard BEIR format
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(split=split)
        
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        
        return corpus, queries, qrels

    def load_documents(self, limit: int | None = None):
        documents = []
        items = list(self.corpus.items())
        
        if limit is not None:
            items = items[:limit]
            
        for document_id, record in items:
            text = record.get("text", "")
            title = record.get("title", "")
            
            if title:
                text = f"{title}\n{text}"
                
            documents.append(
                Document(
                    id=document_id,
                    source=f"beir:{self.name}",
                    text=text,
                    metadata={
                        "benchmark": "BEIR",
                        "dataset": self.name,
                        "benchmark_id": document_id
                    }
                )
            )
        return documents

    def get_queries(self, limit: int | None = None):
        items = list(self.queries.items())
        if limit is not None:
            items = items[:limit]
        return dict(items)

    def get_qrels(self):
        return self.qrels