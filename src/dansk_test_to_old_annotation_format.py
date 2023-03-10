import spacy
from spacy.tokens import DocBin
import random


# Get test_domain.spacy into "old annotation format" (only PER, LOC and ORG)
def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(DocBin().from_disk(f"data/{partition}.spacy").get_docs(nlp.vocab))


def docs_to_old_annotation_format(docs):
    new_docs = []
    for doc in docs:
        new_ents = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORGANIZATION"]:
                new_ents.append(ent)
            if ent.label_ in ["FACILITY", "LOCATION", "GPE"]:
                ent_overwritten = ent
                ent_overwritten.label_ = "LOCATION"
                new_ents.append(ent_overwritten)
        new_doc = doc
        new_doc.ents = new_ents
        new_docs.append(new_doc)
    return new_docs


if __name__ == "__main__":
    domains = [
        "Web",
        "News",
        "Wiki & Books",
        "Legal",
        "dannet",
        "Conversation",
        "Social Media",
        "All Domains",
    ]
    domains = [
        domain.lower().replace(" ", "_").replace("&", "and") for domain in domains
    ]
    domains
    for domain in domains:
        test = load_dansk(f"test_{domain}")
        test_old_format = docs_to_old_annotation_format(test)
        db = DocBin()
        for doc in test_old_format:
            db.add(doc)
        db.to_disk(f"data/test_{domain}_old_format.spacy")
        print(f"data/test_{domain}_old_format.spacy created.\n")
