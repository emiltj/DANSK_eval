import spacy
from spacy.tokens import DocBin
import pandas as pd
import numpy as np
import random


def load_dansk(partition):
    nlp = spacy.blank("da")
    return list(
        DocBin()
        .from_disk(f"data/DANSK_split_by_domain/{partition}.spacy")
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


def convert_output_to_csv(tag_counts_domains, n_docs_domains, n_ents_domains):
    df = pd.DataFrame(tag_counts_domains)
    df["Domain"] = domains
    df["DOCS"] = n_docs_domains
    df["ENTS"] = n_ents_domains
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
domains = [domain.lower().replace(" ", "_").replace("&", "and") for domain in domains]

for p in ["train", "dev", "test"]:

    n_docs_domains = []
    n_ents_domains = []
    tag_counts_domains = []
    for domain in domains:
        docs = load_dansk(f"{p}/{domain}")
        n_docs, n_ents, tag_countss = tag_counts(docs)
        n_docs_domains.append(n_docs)
        n_ents_domains.append(n_ents)
        tag_counts_domains.append(tag_countss)

    outpath = f"output/DANSK_descriptive/{p}/domain_desc_stats.csv"

    df = convert_output_to_csv(
        tag_counts_domains,
        n_docs_domains,
        n_ents_domains,
    )

    save_df_as_csv(df, outpath)


n_docs_domains = []
n_ents_domains = []
tag_counts_domains = []
for domain in domains:
    docs = load_dansk(f"train/{domain}")
    docs.extend(load_dansk(f"dev/{domain}"))
    docs.extend(load_dansk(f"test/{domain}"))
    n_docs, n_ents, tag_countss = tag_counts(docs)
    n_docs_domains.append(n_docs)
    n_ents_domains.append(n_ents)
    tag_counts_domains.append(tag_countss)

outpath = "output/DANSK_descriptive/full/domain_desc_stats.csv"
df = convert_output_to_csv(
    tag_counts_domains,
    n_docs_domains,
    n_ents_domains,
)

save_df_as_csv(df, outpath)

# if __name__ == "__main__":
#     partitions = ["train", "dev", "test"]
#     for p in partitions:
#         docs = load_dansk(p)
#         n_docs, n_ents, partition_tag_counts = tag_counts(docs)
#         print(n_docs)
#         print(n_ents)
#         print(partition_tag_counts)
