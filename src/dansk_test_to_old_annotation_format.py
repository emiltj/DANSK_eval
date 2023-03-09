import spacy
from spacy.tokens import DocBin
import random

# Get test.spacy into "old annotation format" (only PER, LOC and ORG)
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
    test = load_dansk("test")
    test_old_format = docs_to_old_annotation_format(test)
    db = DocBin()
    for doc in test_old_format:
        db.add(doc)
    db.to_disk("data/test_old_format.spacy")
    print("data/test_old_format.spacy created.\n")
