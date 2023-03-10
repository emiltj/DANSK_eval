import spacy
import dacy
import spacy_wrap
import csv, json
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

domain_names = [
    "Web",
    "News",
    "Wiki & Books",
    "Legal",
    "dannet",
    "Conversation",
    "Social Media",
    "All Domains",
]
domain_names = [
    domain.lower().replace(" ", "_").replace("&", "and") for domain in domain_names
]


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(DocBin().from_disk(f"data/{partition}.spacy").get_docs(nlp.vocab))


def retrieve_doc_texts(docs):
    return [d.text for d in docs]


def clean_doc_texts(doc_texts):
    return [text.replace("\n", "") for text in doc_texts]


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
    doc_texts = clean_doc_texts(doc_texts)
    doc_preds = []
    for doc in doc_texts:
        print(doc)
        doc_preds.append(nlp(doc))

    # doc_preds = [nlp(text) for text in doc_texts]
    doc_preds_new_format = docs_to_new_format(doc_preds)

    examples = [
        Example(predicted=doc_preds_new_format[i], reference=gold[i])
        for i in range(len(gold))
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


def define_spacy_models():
    return [
        spacy.load("da_core_news_sm"),
        spacy.load("da_core_news_md"),
        spacy.load("da_core_news_lg"),
    ]


# Defining domain_pairs
domain_data = [
    load_dansk(f"test_{domain_name}_old_format") for domain_name in domain_names
]
domain_pairs = dict(zip(domain_names, domain_data))


# Adding spacy models to model_pairs
spacy_models = define_spacy_models()
model_names = ["da_core_news_sm", "da_core_news_md", "da_core_news_lg"]
model_pairs = dict(zip(model_names, spacy_models))

# Adding saattrupdan/nbailab-base-ner-scandi to model_pairs
saattrupdan_nbailab_base_ner_scandi = nlp = spacy.blank("da")
config = {"model": {"name": "saattrupdan/nbailab-base-ner-scandi"}}
saattrupdan_nbailab_base_ner_scandi.add_pipe(
    "token_classification_transformer", config=config
)
model_pairs["saattrupdan/nbailab-base-ner-scandi"] = saattrupdan_nbailab_base_ner_scandi

# Adding DaCy models:
da_dacy_small_trf = dacy.load("da_dacy_small_trf-0.1.0")
da_dacy_medium_trf = dacy.load("da_dacy_medium_trf-0.1.0")
da_dacy_large_trf = dacy.load("da_dacy_large_trf-0.1.0")
model_pairs["da_dacy_small_trf"] = da_dacy_small_trf
model_pairs["da_dacy_medium_trf"] = da_dacy_medium_trf
# model_pairs["da_dacy_large_trf"] = da_dacy_large_trf


if not model_perf:
    model_perf = {}

for model_name, model in model_pairs.items():
    if model_name not in model_perf.keys():
        print(f'\nMaking predictions using: "{model_name}" on:')
        for domain_name, domain_data in domain_pairs.items():
            print(f" - {domain_name}")
            scores = get_scores(domain_data, model)
            if model_name in model_perf:
                model_perf[model_name][domain_name] = scores
            else:
                model_perf[model_name] = {domain_name: scores}

model_perf


# da_dacy_small_ner_fine_grained = nlp = spacy.blank("da")
# config = {"model": {"name": "emiltj/da_dacy_small_ner_fine_grained"}}
# da_dacy_small_ner_fine_grained.add_pipe(
#     "token_classification_transformer", config=config
# )

# save_perf_as_json(model_perf, "output/performance_scores_old_format.json")
