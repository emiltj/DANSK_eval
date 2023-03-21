import spacy
from spacy.tokens import DocBin
import random


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(DocBin().from_disk(f"data/DANSK/{partition}.spacy").get_docs(nlp.vocab))


def tag_counts(docs):
    # Define list of entity labels included in the docs
    unique_ent_labels = []
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ not in unique_ent_labels:
                unique_ent_labels.append(ent.label_)
    list(set(unique_ent_labels))
    # Define a dictionary with a count of entities in the list of docs
    count_of_ents = {}
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ in count_of_ents:
                count_of_ents[f"{ent.label_}"] += 1

            else:
                count_of_ents[f"{ent.label_}"] = 1
    partition_tag_counts = dict(sorted(count_of_ents.items()))
    n_ents = sum(list(partition_tag_counts.values()))
    n_docs = len(docs)
    return n_docs, n_ents, dict(sorted(count_of_ents.items()))


if __name__ == "__main__":
    partitions = ["train", "dev", "test"]
    for p in partitions:
        docs = load_dansk(p)
        n_docs, n_ents, partition_tag_counts = tag_counts(docs)
        print(n_docs)
        print(n_ents)
        print(partition_tag_counts)
