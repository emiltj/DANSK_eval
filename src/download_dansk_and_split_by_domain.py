from datasets import load_dataset, Dataset
from spacy.tokens import Doc, DocBin
import spacy

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


# Def func for converting Dataset to Doc
def dataset_to_doc(dataset, nlp):
    return [Doc(nlp.vocab).from_json(json_entry) for json_entry in dataset]


def main():
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
    nlp = spacy.blank("da")

    # If the dataset is gated/private, make sure you have run huggingface-cli login
    datasets = load_dataset("chcaa/DANSK")

    # For full test dataset:
    docs_dataset = dataset_to_doc(datasets["test"], nlp)
    db = DocBin()
    outpath = "data/test_all_domains.spacy"
    for doc in docs_dataset:
        db.add(doc)
    db.to_disk(outpath)
    print(f"\n{outpath} has been created.")

    # For test_domains datasets:
    domain_datasets = split_dataset(datasets["test"], domains)
    domains = set(domain_datasets.keys())
    domain_docs = {}
    for domain in domains:
        docs_dataset = dataset_to_doc(domain_datasets[f"{domain}"], nlp)
        domain_docs[f"{domain}"] = docs_dataset

    # Convert to DocBins and save to disk
    for domain in domains:
        db = DocBin()
        domain_outpath_name = domain.replace(" ", "_").lower().replace("&", "and")
        outpath = f"data/test_{domain_outpath_name}.spacy"
        for doc in domain_docs[f"{domain}"]:
            db.add(doc)
        db.to_disk(outpath)
        print(f"\n{outpath} has been created.")


if __name__ == "__main__":
    main()
