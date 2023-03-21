import spacy, json
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

# pip install https://huggingface.co/chcaa/da_dacy_small_ner_fine_grained/resolve/main/da_dacy_small_ner_fine_grained-any-py3-none-any.whl
# pip install https://huggingface.co/chcaa/da_dacy_medium_ner_fine_grained/resolve/main/da_dacy_medium_ner_fine_grained-any-py3-none-any.whl
# pip install https://huggingface.co/chcaa/da_dacy_large_ner_fine_grained/resolve/main/da_dacy_large_ner_fine_grained-any-py3-none-any.whl


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(DocBin().from_disk(f"data/{partition}.spacy").get_docs(nlp.vocab))


def retrieve_doc_texts(docs):
    return [d.text for d in docs]


def get_preds(gold, nlp):
    doc_texts = retrieve_doc_texts(gold)
    return [nlp(doc) for doc in doc_texts]


nlp_ner_small = spacy.load("da_dacy_small_ner_fine_grained")
nlp_ner_medium = spacy.load("da_dacy_medium_ner_fine_grained")
nlp_ner_large = spacy.load("da_dacy_large_ner_fine_grained")

model_names = [
    "da_dacy_small_ner_fine_grained",
    "da_dacy_medium_ner_fine_grained",
    "da_dacy_large_ner_fine_grained",
]
dacy_models = [nlp_ner_small, nlp_ner_medium, nlp_ner_large]
model_pairs = dict(zip(model_names, dacy_models))

test = load_dansk("test_all_domains")
for model_name, model in model_pairs.items():
    model_preds = get_preds(test, model)
    db = DocBin()
    for doc in model_preds:
        db.add(doc)
    size = model_name.split("_")[2]
    outpath = f"output/predictions/{size}_model_preds.spacy"
    db.to_disk(outpath)
    print(f"{outpath} created.\n")

# python src/load_docbin_as_jsonl.py output/predictions/small_model_preds.spacy blank:da --ner > output/predictions/small_model_preds.jsonl
# python src/load_docbin_as_jsonl.py output/predictions/medium_model_preds.spacy blank:da --ner > output/predictions/medium_model_preds.jsonl
# python src/load_docbin_as_jsonl.py output/predictions/large_model_preds.spacy blank:da --ner > output/predictions/large_model_preds.jsonl
