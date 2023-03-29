import spacy, os
from spacy.tokens import DocBin, Doc
from datasets import load_dataset
import shutil

domains = [
    "Web",
    "News",
    "Wiki & Books",
    "Legal",
    "Other",
    "dannet",
    "Conversation",
    "Social Media",
]

# Def func for splitting Dataset up into domains
def split_dataset(dataset, domains):
    domain_datasets = {}
    for domain in domains:
        for doc in dataset:
            if doc["dagw_domain"] == domain:
                if domain not in list(domain_datasets.keys()):
                    domain_datasets[f"{domain}"] = [doc]
                else:
                    domain_datasets[f"{domain}"].append(doc)
    return domain_datasets


def dataset_to_doc(dataset, nlp):
    return [Doc(nlp.vocab).from_json(json_entry) for json_entry in dataset]


def fetch_dansk():
    if not os.path.exists("data/DANSK"):
        os.makedirs("data/DANSK")
    if not os.path.exists("data/DANSK_split_by_domain"):
        os.makedirs("data/DANSK_split_by_domain")
    if not os.path.exists("data/DANSK_split_by_domain/test/"):
        os.makedirs("data/DANSK_split_by_domain/train")
        os.makedirs("data/DANSK_split_by_domain/dev")
        os.makedirs("data/DANSK_split_by_domain/test")

    # Download the datasetdict from the HuggingFace Hub
    try:
        datasets = load_dataset("chcaa/DANSK", cache_dir="cache")
    except FileNotFoundError:
        raise FileNotFoundError(
            "DANSK is not available. It might be due to either HuggingFace being down on the dataset not yet being publically released.",
        )

    full_db = DocBin()
    nlp = spacy.blank("da")
    partitions = ["train", "dev", "test"]
    for p in partitions:
        db = DocBin()
        for doc in [
            Doc(nlp.vocab).from_json(dataset_row) for dataset_row in datasets[f"{p}"]
        ]:
            db.add(doc)
            full_db.add(doc)
        db.to_disk(f"data/DANSK/{p}.spacy")
        db.to_disk(f"data/DANSK_split_by_domain/{p}/all_domains.spacy")

        domains = [
            "Web",
            "News",
            "Wiki & Books",
            "Legal",
            "Other",
            "dannet",
            "Conversation",
            "Social Media",
        ]

        domain_datasets = split_dataset(datasets[p], domains)
        domains = set(domain_datasets.keys())
        domain_docs = {}
        for domain in domains:
            docs_dataset = dataset_to_doc(domain_datasets[f"{domain}"], nlp)
            domain_docs[f"{domain}"] = docs_dataset

        # Convert to DocBins and save to disk

        for domain in domains:
            db = DocBin()
            domain_outpath_name = domain.replace(" ", "_").lower().replace("&", "and")
            outpath = f"data/DANSK_split_by_domain/{p}/{domain_outpath_name}.spacy"
            for doc in domain_docs[f"{domain}"]:
                db.add(doc)
            db.to_disk(outpath)
            print(f"\n{outpath} has been created.")
    full_db.to_disk("data/DANSK/full.spacy")
    shutil.rmtree("./cache")


if __name__ == "__main__":
    fetch_dansk()
