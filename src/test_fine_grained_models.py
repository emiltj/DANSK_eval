import spacy, json
import pandas as pd
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


def save_perf_as_json(performance_scores, outpath):
    # create json object from dictionary
    json_performance_scores = json.dumps(performance_scores)

    with open(f"{outpath}", "w") as f:
        # write json object to file
        f.write(json_performance_scores)


def save_df_as_csv(df, outpath):
    df.to_csv(outpath, sep=",")
    print(f'Saved "{outpath}" succesfully')


nlp_ner_small = spacy.load("da_dacy_small_ner_fine_grained")
nlp_ner_medium = spacy.load("da_dacy_medium_ner_fine_grained")
nlp_ner_large = spacy.load("da_dacy_large_ner_fine_grained")

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

domain_data = [
    load_dansk(f"DANSK_split_by_domain/test/{domain_name}")
    for domain_name in domain_names
]
domain_pairs = dict(zip(domain_names, domain_data))

dacy_models = [nlp_ner_small, nlp_ner_medium, nlp_ner_large]
model_names = [
    "da_dacy_small_ner_fine_grained",
    "da_dacy_medium_ner_fine_grained",
    "da_dacy_large_ner_fine_grained",
]
model_pairs = dict(zip(model_names, dacy_models))

try:
    model_perf
except:
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

save_perf_as_json(model_perf, "output/ner_models_performance/new_format_all_stats.json")


# Get F1-scores for each model for each domain
df = pd.DataFrame.from_records(
    [
        (level1, level2, level3, leaf)
        for level1, level2_dict in model_perf.items()
        for level2, level3_dict in level2_dict.items()
        for level3, leaf in level3_dict.items()
    ],
    columns=["Model", "Domain", "Metric", "Score"],
)
df_no_individual_tags = df[pd.to_numeric(df["Score"], errors="coerce").notnull()]

df_f1_no_individual_tags = df_no_individual_tags[
    df_no_individual_tags["Metric"] == "ents_f"
]

domain_f1 = pd.pivot(
    df_f1_no_individual_tags,
    index=["Model", "Metric"],
    columns="Domain",
    values="Score",
)
domain_f1.rename(columns={"all_domains": "across_domains"}, inplace=True)

domain_f1 = domain_f1.reset_index()
domain_f1 = domain_f1.rename_axis(None, axis=1)
domain_f1 = domain_f1.set_index("Model")
domain_f1 = domain_f1.drop(["Metric"], axis=1)
domain_f1
domain_f1_transposed = domain_f1.T
domain_f1_transposed
save_df_as_csv(domain_f1, "output/ner_models_performance/domain_f1_wide.csv")
save_df_as_csv(domain_f1_transposed, "output/ner_models_performance/domain_f1_long.csv")


# Get F1-scores for each model for each tags (across domains)
df = pd.DataFrame.from_records(
    [
        (level1, level2, level3, level4, level5, leaf)
        for level1, level2_dict in model_perf.items()
        for level2, level3_dict in level2_dict.items()
        for level3, level4_dict in level3_dict.items()
        if type(level4_dict) == dict
        for level4, level5_dict in level4_dict.items()
        for level5, leaf in level5_dict.items()
    ],
    columns=["Model", "Domain", "S", "Tag", "Metric", "Score"],
)
df = df[df["Metric"] == "f"]
df = df[["Model", "Domain", "Tag", "Score"]]
tag_f1 = df[df["Domain"] == "all_domains"]
tag_f1.rename(columns={"Score": "F1-score"}, inplace=True)
tag_f1 = tag_f1.sort_values(["Model", "Tag"], ascending=[True, True])
tag_f1_wide = pd.pivot(
    tag_f1,
    index=["Model", "Domain"],
    columns="Tag",
    values="F1-score",
)

save_df_as_csv(tag_f1_wide, "output/ner_models_performance/tag_f1_wide.csv")
save_df_as_csv(tag_f1, "output/ner_models_performance/tag_f1.csv")
