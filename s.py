import spacy, json, os
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

os.chdir("/Users/emiltrencknerjessen/Desktop/priv/DaCy/training/ner_fine_grained")


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


def retrieve_doc_texts(docs):
    return [d.text for d in docs]


def get_scores(gold, nlp):
    doc_texts = retrieve_doc_texts(gold)
    doc_preds = []
    for doc in doc_texts:
        doc_preds.append(nlp(doc))
    examples = [
        Example(predicted=doc_preds[i], reference=gold[i]) for i in range(len(gold))
    ]

    scorer = Scorer()
    scores = scorer.score(examples)

    scores = {k: scores[k] for k in ("ents_p", "ents_r", "ents_f", "ents_per_type")}
    return scores


nlp = spacy.blank("da")
test = list(DocBin().from_disk("corpus/test.spacy").get_docs(nlp.vocab))
dev = list(DocBin().from_disk("corpus/dev.spacy").get_docs(nlp.vocab))

nlp_ner_small = spacy.load("da_dacy_small_ner_fine_grained")
nlp_ner_medium = spacy.load("da_dacy_medium_ner_fine_grained")
nlp_ner_large = spacy.load("da_dacy_large_ner_fine_grained")


docs = [
    nlp_ner_small("Dette er en tekst om Aarhus kl 17:30"),
    nlp_ner_small("Dette er en tekst om Aarhus Hovedbanegård og Danmark"),
]


for doc in docs:
    for ent in doc.ents:
        print(ent.text)
        print(ent.label_)

new_docs = docs_to_old_annotation_format(docs)

for doc in new_docs:
    for ent in doc.ents:
        print(ent.text)
        print(ent.label_)


sm_dev = get_scores(dev, nlp_ner_small)
sm_test = get_scores(test, nlp_ner_small)
md_dev = get_scores(dev, nlp_ner_medium)
md_test = get_scores(test, nlp_ner_medium)
lg_dev = get_scores(dev, nlp_ner_large)
lg_test = get_scores(test, nlp_ner_large)

lg_dev
lg_test

sm_dev = {
    "ents_p": 0.7811653116531165,
    "ents_r": 0.7702070808283233,
    "ents_f": 0.7756474941136898,
}

sm_test = {
    "ents_p": 0.7791728212703102,
    "ents_r": 0.7950263752825923,
    "ents_f": 0.7870197687430063,
}

md_dev = {
    "ents_p": 0.7937743190661478,
    "ents_r": 0.8176352705410822,
    "ents_f": 0.8055281342546889,
}

md_test = {
    "ents_p": 0.776824034334764,
    "ents_r": 0.8183873398643556,
    "ents_f": 0.7970642201834862,
}

lg_dev = {
    "ents_p": 0.8130293159609121,
    "ents_r": 0.8336673346693386,
    "ents_f": 0.8232189973614776,
}

lg_test = {
    "ents_p": 0.7761194029850746,
    "ents_r": 0.8229088168801808,
    "ents_f": 0.7988295537673737,
}
