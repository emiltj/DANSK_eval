import spacy
from spacy.tokens import DocBin
import pandas as pd
import numpy as np
import random
import os


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(
        DocBin()
        .from_disk(f"data/DANSK_split_by_source/{partition}.spacy")
        .get_docs(nlp.vocab)
    )


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


def convert_output_to_csv(tag_counts_sources, n_docs_sources, n_ents_sources):
    df = pd.DataFrame(tag_counts_sources)
    df["Source"] = sources
    df["DOCS"] = n_docs_sources
    df["ENTS"] = n_ents_sources
    for _ in range(3):
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
    df = df.fillna(1000000)
    type_dict = {f"{c}": "int" for c in df.columns[1:]}
    df = df.astype(type_dict, errors="ignore")
    df = df.replace(1000000, "N/A")
    return df


def save_df_as_csv(df, outpath):
    df.to_csv(outpath, sep=",")
    print(f'Saved "{outpath}" succesfully')


sources = [
    "retsinformationdk",
    "skat",
    "retspraksis",
    "hest",
    "cc",
    "adl",
    # "botxt",
    "danavis",
    "dannet",
    # "depbank",
    "ep",
    "ft",
    # "gutenberg",
    # "jvj",
    "naat",
    "opensub",
    # "relig",
    "spont",
    # "synne",
    "tv2r",
    "wiki",
    "wikibooks",
    "wikisource",
    # "twfv19",
]

source_extended_mapping = {
    "retsinformationdk": "retsinformation.dk (Danish legal information)",
    "skat": "Skat (Danish tax authority)",
    "retspraksis": "retspraksis (Danish legal information)",
    "hest": "Hestenettet (Danish debate forum)",
    "cc": "Common Crawl",
    "adl": " Archive for Danish Literature",
    # "botxt": "Bornholmsk (Danish dialect)",
    "danavis": "Danish daily newspapers",
    "dannet": "DanNet (Danish WordNet)",
    # "depbank": "Danish Dependency Treebank",
    "ep": "European Parliament",
    "ft": "Folketinget (Danish Parliament)",
    # "gutenberg": "Gutenberg",
    # "jvj": "Johannes V. Jensen (Danish poet)",
    "naat": "NAAT",
    "opensub": "Open Subtitles",
    # "relig": "Religious texts",
    "spont": "Spontaneous speech",
    # "synne": "Synderjysk (Danish dialect)",
    "tv2r": "TV 2 Radio (Danish news)",
    "wiki": "Wikipedia",
    "wikibooks": "Wikibooks",
    "wikisource": "Wikisource",
    # "twfv19": "Twitter Folketingsvalget 2019 (Danish election tweets)",
}

sources = [source.lower().replace(" ", "_").replace("&", "and") for source in sources]


for p in ["train", "dev", "test"]:

    n_docs_sources = []
    n_ents_sources = []
    tag_counts_sources = []
    for source in sources:
        docs = load_dansk(f"{p}/{source}")
        n_docs, n_ents, tag_countss = tag_counts(docs)
        n_docs_sources.append(n_docs)
        n_ents_sources.append(n_ents)
        tag_counts_sources.append(tag_countss)

    df = convert_output_to_csv(
        tag_counts_sources,
        n_docs_sources,
        n_ents_sources,
    )

    df_wide = df.set_index("Source")
    df_wide = df_wide.T

    save_df_as_csv(df, f"output/DANSK_descriptive/{p}/source_desc_stats.csv")
    save_df_as_csv(df_wide, f"output/DANSK_descriptive/{p}/source_desc_stats_wide.csv")


n_docs_sources = []
n_ents_sources = []
tag_counts_sources = []
for source in sources:
    docs = load_dansk(f"train/{source}")
    docs.extend(load_dansk(f"dev/{source}"))
    docs.extend(load_dansk(f"test/{source}"))
    n_docs, n_ents, tag_countss = tag_counts(docs)
    n_docs_sources.append(n_docs)
    n_ents_sources.append(n_ents)
    tag_counts_sources.append(tag_countss)


df = convert_output_to_csv(
    tag_counts_sources,
    n_docs_sources,
    n_ents_sources,
)

df_wide = df.set_index("Source")
df_wide = df_wide.T

save_df_as_csv(df, "output/DANSK_descriptive/full/source_desc_stats.csv")
save_df_as_csv(df_wide, "output/DANSK_descriptive/full/source_desc_stats_wide.csv")
