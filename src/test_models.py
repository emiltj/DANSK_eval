import spacy
import json, dacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
import pandas as pd

# pip install https://huggingface.co/chcaa/da_dacy_small_ner_fine_grained/resolve/main/da_dacy_small_ner_fine_grained-any-py3-none-any.whl
# pip install https://huggingface.co/chcaa/da_dacy_medium_ner_fine_grained/resolve/main/da_dacy_medium_ner_fine_grained-any-py3-none-any.whl
# pip install https://huggingface.co/chcaa/da_dacy_large_ner_fine_grained/resolve/main/da_dacy_large_ner_fine_grained-any-py3-none-any.whl
# python -m spacy download da_core_news_sm
# python -m spacy download da_core_news_md
# python -m spacy download da_core_news_lg


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


def get_scores(gold, nlp, model_name):
    doc_texts = retrieve_doc_texts(gold)
    # doc_texts = clean_doc_texts(doc_texts)
    doc_preds = []
    bad_texts_indices = []
    for i, doc in enumerate(doc_texts):
        try:
            doc_preds.append(nlp(doc))
        except:
            bad_texts_indices.append(i)
    if model_name in new_format_output_models:
        doc_preds_new_format = docs_to_old_annotation_format(doc_preds)
    if model_name not in new_format_output_models:
        doc_preds_new_format = docs_to_new_format(doc_preds)
    bad_texts_indices.sort(reverse=True)
    for i in bad_texts_indices:
        gold.pop(i)

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


def save_df_as_csv(df, outpath):
    df.to_csv(outpath, sep=",")
    print(f'Saved "{outpath}" succesfully')


def define_spacy_models():
    return [
        spacy.load("da_core_news_sm"),
        spacy.load("da_core_news_md"),
        spacy.load("da_core_news_lg"),
    ]


# Defining domain_pairs
domain_data = [
    load_dansk(f"DANSK_split_by_domain/test/{domain_name}_old_format")
    for domain_name in domain_names
]
domain_pairs = dict(zip(domain_names, domain_data))

# Adding spacy models to model_pairs
spacy_models = define_spacy_models()
model_names = ["da_core_news_sm", "da_core_news_md", "da_core_news_lg"]
model_pairs = dict(zip(model_names, spacy_models))

# Adding new spacy models to model_pairs
da_dacy_small_ner_fine_grained = spacy.load("da_dacy_small_ner_fine_grained")
da_dacy_medium_ner_fine_grained = spacy.load("da_dacy_medium_ner_fine_grained")
da_dacy_large_ner_fine_grained = spacy.load("da_dacy_large_ner_fine_grained")
model_pairs["da_dacy_small_ner_fine_grained"] = da_dacy_small_ner_fine_grained
model_pairs["da_dacy_medium_ner_fine_grained"] = da_dacy_medium_ner_fine_grained
model_pairs["da_dacy_large_ner_fine_grained"] = da_dacy_large_ner_fine_grained

new_format_output_models = [
    "da_dacy_small_ner_fine_grained",
    "da_dacy_medium_ner_fine_grained",
    "da_dacy_large_ner_fine_grained",
]

saattrupdan_nbailab_base_ner_scandi = dacy.load(
    "da_dacy_small_trf-0.1.0", exclude=["ner"]
)
saattrupdan_nbailab_base_ner_scandi.add_pipe("dacy/ner")
model_pairs["saattrupdan/nbailab-base-ner-scandi"] = saattrupdan_nbailab_base_ner_scandi


# Adding DaCy models:
da_dacy_small_trf = dacy.load("da_dacy_small_trf-0.1.0")
model_pairs["da_dacy_small_trf"] = da_dacy_small_trf

da_dacy_medium_trf = dacy.load("da_dacy_medium_trf-0.1.0")
model_pairs["da_dacy_medium_trf"] = da_dacy_medium_trf

da_dacy_large_trf = dacy.load("da_dacy_large_trf-0.1.0")
model_pairs["da_dacy_large_trf"] = da_dacy_large_trf


try:
    model_perf
except:
    model_perf = {}

for model_name, model in model_pairs.items():
    if model_name not in model_perf.keys():
        print(f'\nMaking predictions using: "{model_name}" on:')
        for domain_name, domain_data in domain_pairs.items():
            print(f" - {domain_name}")
            scores = get_scores(domain_data, model, model_name)
            if model_name in model_perf:
                model_perf[model_name][domain_name] = scores
            else:
                model_perf[model_name] = {domain_name: scores}


# Save performance
with open("output/other_models_performance/old_format_all_stats.json", "w") as fp:
    json.dump(model_perf, fp)

# Open performance
with open("output/other_models_performance/old_format_all_stats.json", "r") as j:
    data = json.loads(j.read())


# Get F1-scores for each model for each domain
df = pd.DataFrame.from_records(
    [
        (level1, level2, level3, leaf)
        for level1, level2_dict in data.items()  # MAY CHANGE TO MODEL_PERF
        for level2, level3_dict in level2_dict.items()
        for level3, leaf in level3_dict.items()
    ],
    columns=["Model", "Domain", "Metric", "Score"],
)


df_no_individual_tags = df[pd.to_numeric(df["Score"], errors="coerce").notnull()]

df_across = df_no_individual_tags[df_no_individual_tags["Domain"] == "all_domains"]


df_f_r_p_with_name = pd.pivot(
    df_across,
    index=["Model"],
    columns="Metric",
    values="Score",
)
df_f_p_r = pd.DataFrame(
    {
        "F1": df_f_r_p_with_name["ents_f"],
        "Recall": df_f_r_p_with_name["ents_r"],
        "Precision": df_f_r_p_with_name["ents_p"],
    }
)

reorder_list = [
    "da_dacy_large_ner_fine_grained",
    "da_dacy_medium_ner_fine_grained",
    "da_dacy_small_ner_fine_grained",
    "saattrupdan/nbailab-base-ner-scandi",
    "da_dacy_large_trf",
    "da_dacy_medium_trf",
    "da_dacy_small_trf",
    "da_core_news_lg",
    "da_core_news_md",
    "da_core_news_sm",
]

df_f_p_r = df_f_p_r.reindex(reorder_list)
df_f_p_r = df_f_p_r.astype(float).round(3)
df_f_p_r = df_f_p_r.astype(float)
save_df_as_csv(df_f_p_r, "output/other_models_performance/f1_recall_precision.csv")

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
domain_f1 = domain_f1.sort_values(by=["Model"])
domain_f1 = domain_f1.reset_index()
domain_f1 = domain_f1.rename_axis(None, axis=1)
domain_f1 = domain_f1.set_index("Model")
domain_f1 = domain_f1.drop(["Metric"], axis=1)
domain_f1_transposed = domain_f1.T


reorder_list_index = [3, 5, 7, 4, 6, 8, 0, 1, 2, 8]

domain_f1_transposed

domain_f1_transposed = domain_f1_transposed[
    domain_f1_transposed.columns[reorder_list_index]
]
domain_f1 = domain_f1.reindex(reorder_list)

save_df_as_csv(domain_f1, "output/other_models_performance/domain_f1_long.csv")
save_df_as_csv(
    domain_f1_transposed, "output/other_models_performance/domain_f1_wide.csv"
)


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

reorder_list = [
    "da_dacy_large_ner_fine_grained",
    "da_dacy_medium_ner_fine_grained",
    "da_dacy_small_ner_fine_grained",
    "da_dacy_large_trf",
    "da_dacy_medium_trf",
    "da_dacy_small_trf",
    "da_core_news_lg",
    "da_core_news_md",
    "da_core_news_sm",
    "saattrupdan/nbailab-base-ner-scandi",
]
reorder_list_index = [3, 5, 7, 4, 6, 8, 0, 1, 2, 8]

tag_f1_wide = tag_f1_wide.reset_index()
tag_f1_wide = tag_f1_wide.set_index("Model")
tag_f1_wide = tag_f1_wide.reindex(reorder_list)
tag_f1_wide = tag_f1_wide[["LOCATION", "ORGANIZATION", "PERSON"]]

save_df_as_csv(tag_f1_wide, "output/other_models_performance/tag_f1_wide.csv")
save_df_as_csv(tag_f1, "output/other_models_performance/tag_f1_long.csv")
