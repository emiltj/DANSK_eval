import spacy
import dacy
import spacy_wrap
import csv, json
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

performance_scores = {}


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(DocBin().from_disk(f"data/{partition}.spacy").get_docs(nlp.vocab))


def retrieve_doc_texts(docs):
    return [d.text for d in docs]


# Removes "MISC" annotations and changes "PER" to "PERSON", etc.
def docs_to_new_format(docs):
    new_docs = []
    for doc in docs:
        new_ents = []
        for ent in doc.ents:
            if ent.label_ == "PER":
                ent_overwritten = ent
                ent_overwritten.label_ = "PERSON"
                new_ents.append(ent_overwritten)
            if ent.label_ == "ORG":
                ent_overwritten = ent
                ent_overwritten.label_ = "ORGANIZATION"
                new_ents.append(ent_overwritten)
            if ent.label_ == "LOC":
                ent_overwritten = ent
                ent_overwritten.label_ = "LOCATION"
                new_ents.append(ent_overwritten)
        new_doc = doc
        new_doc.ents = new_ents
        new_docs.append(new_doc)
    return new_docs


def get_scores(gold, nlp):
    doc_texts = retrieve_doc_texts(gold)
    doc_preds = [nlp(text) for text in doc_texts]
    doc_preds_new_format = docs_to_new_format(doc_preds)

    examples = [
        Example(predicted=doc_preds_new_format[i], reference=docs[i])
        for i in range(len(docs))
    ]

    scorer = Scorer()
    scores = scorer.score(examples)
    scores = {k: scores[k] for k in ("ents_p", "ents_r", "ents_f", "ents_per_type")}
    return scores


def save_perf_as_json(performance_scores, outpath):
    # create json object from dictionary
    json_performance_scores = json.dumps(performance_scores)

    with open(f"{outpath}", "w") as f:
        # write json object to file
        f.write(json_performance_scores)


saattrupdan_nbailab_base_ner_scandi = nlp = spacy.blank("da")
config = {"model": {"name": "saattrupdan/nbailab-base-ner-scandi"}}
saattrupdan_nbailab_base_ner_scandi.add_pipe(
    "token_classification_transformer", config=config
)

da_dacy_small_ner_fine_grained = nlp = spacy.blank("da")
config = {"model": {"name": "emiltj/da_dacy_small_ner_fine_grained"}}
da_dacy_small_ner_fine_grained.add_pipe(
    "token_classification_transformer", config=config
)

da_dacy_small_trf = dacy.load("da_dacy_small_trf-0.1.0")
da_dacy_medium_trf = dacy.load("da_dacy_medium_trf-0.1.0")
# da_dacy_large_trf = dacy.load("da_dacy_large_trf-0.1.0") # Causes error. Ask Kenneth, maybe? Sent error to myself on Slack

da_core_news_sm = spacy.load("da_core_news_sm")
da_core_news_md = spacy.load("da_core_news_md")
da_core_news_lg = spacy.load("da_core_news_lg")

models = {
    "da_dacy_small_trf": da_dacy_small_trf,  # working
    "da_dacy_medium_trf": da_dacy_medium_trf,  # working
    # "da_dacy_large_trf" : da_dacy_large_trf, # NOT working
    "da_core_news_sm": da_core_news_sm,  # working
    "da_core_news_md": da_core_news_md,  # working
    "da_core_news_lg": da_core_news_lg,  # working
    # "saattrupdan_nbailab_base_ner_scandi": saattrupdan_nbailab_base_ner_scandi, # Not working
    # "da_dacy_small_ner_fine_grained": da_dacy_small_ner_fine_grained, # not working
}


docs = load_dansk("test_old_format")
for name, nlp in models.items():
    if name not in performance_scores.keys():
        print(f"\nAnnotating using: {name} ...")
        performance_scores[f"{name}"] = get_scores(docs, nlp)

save_perf_as_json(performance_scores, "output/performance_scores.json")

for name, scores in performance_scores.items():
    print(name)
    print(scores["ents_f"])
