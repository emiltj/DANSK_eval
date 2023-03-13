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


# def clean_doc_texts(doc_texts):
#     return [
#         text.replace("\n", "").replace("‘", "'").replace("’", "'").replace("”", '"')
#         for text in doc_texts
#     ]


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
    # doc_texts = clean_doc_texts(doc_texts)
    doc_preds = []
    bad_texts_indices = []
    for i, doc in enumerate(doc_texts):
        try:
            doc_preds.append(nlp(doc))
        except:
            bad_texts_indices.append(i)

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
# saattrupdan_nbailab_base_ner_scandi = nlp = spacy.blank("da")
# config = {"model": {"name": "saattrupdan/nbailab-base-ner-scandi"}}
# saattrupdan_nbailab_base_ner_scandi.add_pipe(
#     "token_classification_transformer", config=config
# )
# model_pairs["saattrupdan/nbailab-base-ner-scandi"] = saattrupdan_nbailab_base_ner_scandi

################### GOTTEN TO HERE ###################

# # Adding DaCy models:
# da_dacy_small_trf = dacy.load("da_dacy_small_trf-0.1.0")
# model_pairs["da_dacy_small_trf"] = da_dacy_small_trf

# da_dacy_medium_trf = dacy.load("da_dacy_medium_trf-0.1.0")
# model_pairs["da_dacy_medium_trf"] = da_dacy_medium_trf

# da_dacy_large_trf = spacy.load("da_dacy_large_trf")
# da_dacy_large_trf = dacy.load("da_dacy_large_trf-0.1.0")
# model_pairs["da_dacy_large_trf"] = da_dacy_large_trf

################### GOTTEN TO HERE ###################

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

for i in model_perf.keys():
    print(i)

# da_dacy_small_ner_fine_grained = nlp = spacy.blank("da")
# config = {"model": {"name": "emiltj/da_dacy_small_ner_fine_grained"}}
# da_dacy_small_ner_fine_grained.add_pipe(
#     "token_classification_transformer", config=config
# )

# save_perf_as_json(model_perf, "output/performance_scores_old_format.json")

# data = json.loads("output/performance_scores_old_format_all_models.json")

model_perf = {
    "da_dacy_large_trf": {
        "web": {
            "ents_p": 0.4946524064171123,
            "ents_r": 0.45232273838630804,
            "ents_f": 0.4725415070242656,
            "ents_per_type": {
                "PERSON": {"p": 0.5, "r": 0.5408163265306123, "f": 0.5196078431372549},
                "LOCATION": {
                    "p": 0.6291390728476821,
                    "r": 0.6506849315068494,
                    "f": 0.6397306397306397,
                },
                "ORGANIZATION": {
                    "p": 0.3162393162393162,
                    "r": 0.22424242424242424,
                    "f": 0.2624113475177305,
                },
            },
        },
        "news": {
            "ents_p": 0.7,
            "ents_r": 0.5833333333333334,
            "ents_f": 0.6363636363636365,
            "ents_per_type": {
                "PERSON": {"p": 0.5, "r": 0.375, "f": 0.42857142857142855},
                "LOCATION": {
                    "p": 0.7272727272727273,
                    "r": 0.7272727272727273,
                    "f": 0.7272727272727273,
                },
                "ORGANIZATION": {"p": 1.0, "r": 0.6, "f": 0.7499999999999999},
            },
        },
        "wiki_and_books": {
            "ents_p": 0.4050632911392405,
            "ents_r": 0.6808510638297872,
            "ents_f": 0.5079365079365079,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.3333333333333333,
                    "r": 0.3333333333333333,
                    "f": 0.3333333333333333,
                },
                "PERSON": {
                    "p": 0.3548387096774194,
                    "r": 0.6470588235294118,
                    "f": 0.4583333333333333,
                },
                "LOCATION": {
                    "p": 0.46153846153846156,
                    "r": 0.8571428571428571,
                    "f": 0.6,
                },
            },
        },
        "legal": {
            "ents_p": 0.5476190476190477,
            "ents_r": 0.5287356321839081,
            "ents_f": 0.5380116959064327,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.5909090909090909,
                    "r": 0.49056603773584906,
                    "f": 0.5360824742268041,
                },
                "PERSON": {
                    "p": 0.5333333333333333,
                    "r": 0.5517241379310345,
                    "f": 0.5423728813559322,
                },
                "LOCATION": {"p": 0.4, "r": 0.8, "f": 0.5333333333333333},
            },
        },
        "dannet": {
            "ents_p": 0.0,
            "ents_r": 0.0,
            "ents_f": 0.0,
            "ents_per_type": {"LOCATION": {"p": 0.0, "r": 0.0, "f": 0.0}},
        },
        "conversation": {
            "ents_p": 0.5135135135135135,
            "ents_r": 0.3877551020408163,
            "ents_f": 0.441860465116279,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.45454545454545453,
                    "r": 0.2631578947368421,
                    "f": 0.3333333333333333,
                },
                "LOCATION": {
                    "p": 0.5263157894736842,
                    "r": 0.5263157894736842,
                    "f": 0.5263157894736842,
                },
                "ORGANIZATION": {
                    "p": 0.5714285714285714,
                    "r": 0.36363636363636365,
                    "f": 0.4444444444444444,
                },
            },
        },
        "social_media": {
            "ents_p": 0.47368421052631576,
            "ents_r": 0.5625,
            "ents_f": 0.5142857142857142,
            "ents_per_type": {
                "ORGANIZATION": {"p": 0.5, "r": 0.3333333333333333, "f": 0.4},
                "PERSON": {"p": 0.5, "r": 0.75, "f": 0.6},
                "LOCATION": {"p": 0.3333333333333333, "r": 0.5, "f": 0.4},
            },
        },
        "all_domains": {
            "ents_p": 0.500805152979066,
            "ents_r": 0.48142414860681115,
            "ents_f": 0.49092344119968434,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.4925373134328358,
                    "r": 0.518324607329843,
                    "f": 0.5051020408163266,
                },
                "LOCATION": {
                    "p": 0.5805084745762712,
                    "r": 0.6650485436893204,
                    "f": 0.6199095022624435,
                },
                "ORGANIZATION": {
                    "p": 0.4076086956521739,
                    "r": 0.30120481927710846,
                    "f": 0.3464203233256351,
                },
            },
        },
    },
    "saattrupdan/nbailab-base-ner-scandi": {
        "web": {
            "ents_p": 0.6781002638522428,
            "ents_r": 0.628361858190709,
            "ents_f": 0.6522842639593909,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.7363636363636363,
                    "r": 0.826530612244898,
                    "f": 0.7788461538461539,
                },
                "LOCATION": {
                    "p": 0.7394366197183099,
                    "r": 0.7191780821917808,
                    "f": 0.7291666666666666,
                },
                "ORGANIZATION": {
                    "p": 0.5590551181102362,
                    "r": 0.4303030303030303,
                    "f": 0.4863013698630137,
                },
            },
        },
        "news": {
            "ents_p": 0.9473684210526315,
            "ents_r": 0.8571428571428571,
            "ents_f": 0.9,
            "ents_per_type": {
                "PERSON": {"p": 1.0, "r": 0.8571428571428571, "f": 0.923076923076923},
                "LOCATION": {
                    "p": 0.9090909090909091,
                    "r": 0.9090909090909091,
                    "f": 0.9090909090909091,
                },
                "ORGANIZATION": {"p": 1.0, "r": 0.6666666666666666, "f": 0.8},
            },
        },
        "wiki_and_books": {
            "ents_p": 0.7111111111111111,
            "ents_r": 0.8205128205128205,
            "ents_f": 0.7619047619047619,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.5555555555555556,
                    "r": 0.7142857142857143,
                    "f": 0.6250000000000001,
                },
                "LOCATION": {"p": 0.76, "r": 0.95, "f": 0.8444444444444444},
                "PERSON": {
                    "p": 0.7272727272727273,
                    "r": 0.6666666666666666,
                    "f": 0.6956521739130435,
                },
            },
        },
        "legal": {
            "ents_p": 0.671875,
            "ents_r": 0.5512820512820513,
            "ents_f": 0.6056338028169015,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.7575757575757576,
                    "r": 0.4807692307692308,
                    "f": 0.5882352941176471,
                },
                "PERSON": {"p": 0.56, "r": 0.6666666666666666, "f": 0.6086956521739131},
                "LOCATION": {
                    "p": 0.6666666666666666,
                    "r": 0.8,
                    "f": 0.7272727272727272,
                },
            },
        },
        "dannet": {
            "ents_p": 1.0,
            "ents_r": 1.0,
            "ents_f": 1.0,
            "ents_per_type": {"LOCATION": {"p": 1.0, "r": 1.0, "f": 1.0}},
        },
        "conversation": {
            "ents_p": 0.6857142857142857,
            "ents_r": 0.4897959183673469,
            "ents_f": 0.5714285714285715,
            "ents_per_type": {
                "LOCATION": {
                    "p": 0.8947368421052632,
                    "r": 0.8947368421052632,
                    "f": 0.8947368421052632,
                },
                "PERSON": {
                    "p": 0.3076923076923077,
                    "r": 0.21052631578947367,
                    "f": 0.25,
                },
                "ORGANIZATION": {
                    "p": 1.0,
                    "r": 0.2727272727272727,
                    "f": 0.42857142857142855,
                },
            },
        },
        "social_media": {
            "ents_p": 0.45454545454545453,
            "ents_r": 0.3333333333333333,
            "ents_f": 0.3846153846153846,
            "ents_per_type": {
                "ORGANIZATION": {"p": 0.0, "r": 0.0, "f": 0.0},
                "PERSON": {
                    "p": 0.6666666666666666,
                    "r": 0.5714285714285714,
                    "f": 0.6153846153846153,
                },
                "LOCATION": {"p": 0.25, "r": 0.5, "f": 0.3333333333333333},
            },
        },
        "all_domains": {
            "ents_p": 0.6864864864864865,
            "ents_r": 0.612540192926045,
            "ents_f": 0.6474086661002548,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.6842105263157895,
                    "r": 0.6763005780346821,
                    "f": 0.680232558139535,
                },
                "LOCATION": {
                    "p": 0.7559808612440191,
                    "r": 0.7707317073170732,
                    "f": 0.7632850241545894,
                },
                "ORGANIZATION": {
                    "p": 0.6057142857142858,
                    "r": 0.4344262295081967,
                    "f": 0.5059665871121719,
                },
            },
        },
    },
    "da_dacy_small_trf": {
        "web": {
            "ents_p": 0.5483870967741935,
            "ents_r": 0.4572127139364303,
            "ents_f": 0.4986666666666666,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.5426356589147286,
                    "r": 0.7142857142857143,
                    "f": 0.6167400881057268,
                },
                "LOCATION": {
                    "p": 0.6637931034482759,
                    "r": 0.5273972602739726,
                    "f": 0.5877862595419847,
                },
                "ORGANIZATION": {
                    "p": 0.4166666666666667,
                    "r": 0.24242424242424243,
                    "f": 0.3065134099616858,
                },
            },
        },
        "news": {
            "ents_p": 0.75,
            "ents_r": 0.7142857142857143,
            "ents_f": 0.7317073170731706,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.3333333333333333,
                    "r": 0.3333333333333333,
                    "f": 0.3333333333333333,
                },
                "PERSON": {
                    "p": 0.8333333333333334,
                    "r": 0.7142857142857143,
                    "f": 0.7692307692307692,
                },
                "LOCATION": {
                    "p": 0.8181818181818182,
                    "r": 0.8181818181818182,
                    "f": 0.8181818181818182,
                },
            },
        },
        "wiki_and_books": {
            "ents_p": 0.4166666666666667,
            "ents_r": 0.6410256410256411,
            "ents_f": 0.5050505050505051,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.15384615384615385,
                    "r": 0.2857142857142857,
                    "f": 0.2,
                },
                "PERSON": {
                    "p": 0.46153846153846156,
                    "r": 0.5,
                    "f": 0.48000000000000004,
                },
                "LOCATION": {"p": 0.5, "r": 0.85, "f": 0.6296296296296295},
            },
        },
        "legal": {
            "ents_p": 0.5263157894736842,
            "ents_r": 0.7692307692307693,
            "ents_f": 0.625,
            "ents_per_type": {
                "LOCATION": {"p": 0.4, "r": 0.8, "f": 0.5333333333333333},
                "ORGANIZATION": {"p": 0.5, "r": 0.75, "f": 0.6},
                "PERSON": {
                    "p": 0.6538461538461539,
                    "r": 0.8095238095238095,
                    "f": 0.7234042553191489,
                },
            },
        },
        "dannet": {
            "ents_p": 0.5,
            "ents_r": 1.0,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "LOCATION": {"p": 0.5, "r": 1.0, "f": 0.6666666666666666}
            },
        },
        "conversation": {
            "ents_p": 0.7586206896551724,
            "ents_r": 0.4489795918367347,
            "ents_f": 0.564102564102564,
            "ents_per_type": {
                "LOCATION": {"p": 1.0, "r": 0.9473684210526315, "f": 0.972972972972973},
                "PERSON": {
                    "p": 0.16666666666666666,
                    "r": 0.05263157894736842,
                    "f": 0.08,
                },
                "ORGANIZATION": {
                    "p": 0.6,
                    "r": 0.2727272727272727,
                    "f": 0.37499999999999994,
                },
            },
        },
        "social_media": {
            "ents_p": 0.3333333333333333,
            "ents_r": 0.3333333333333333,
            "ents_f": 0.3333333333333333,
            "ents_per_type": {
                "ORGANIZATION": {"p": 0.25, "r": 0.16666666666666666, "f": 0.2},
                "PERSON": {
                    "p": 0.3333333333333333,
                    "r": 0.42857142857142855,
                    "f": 0.375,
                },
                "LOCATION": {"p": 0.5, "r": 0.5, "f": 0.5},
            },
        },
        "all_domains": {
            "ents_p": 0.5409556313993175,
            "ents_r": 0.5096463022508039,
            "ents_f": 0.5248344370860928,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.5392670157068062,
                    "r": 0.5953757225433526,
                    "f": 0.5659340659340659,
                },
                "LOCATION": {
                    "p": 0.6564102564102564,
                    "r": 0.624390243902439,
                    "f": 0.64,
                },
                "ORGANIZATION": {
                    "p": 0.43,
                    "r": 0.3524590163934426,
                    "f": 0.38738738738738737,
                },
            },
        },
    },
    "da_dacy_medium_trf": {
        "web": {
            "ents_p": 0.6148648648648649,
            "ents_r": 0.6674816625916871,
            "ents_f": 0.6400937866354045,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.6532258064516129,
                    "r": 0.826530612244898,
                    "f": 0.7297297297297297,
                },
                "LOCATION": {
                    "p": 0.7197452229299363,
                    "r": 0.773972602739726,
                    "f": 0.7458745874587459,
                },
                "ORGANIZATION": {
                    "p": 0.48466257668711654,
                    "r": 0.47878787878787876,
                    "f": 0.4817073170731707,
                },
            },
        },
        "news": {
            "ents_p": 0.9,
            "ents_r": 0.8571428571428571,
            "ents_f": 0.8780487804878048,
            "ents_per_type": {
                "PERSON": {"p": 1.0, "r": 0.8571428571428571, "f": 0.923076923076923},
                "LOCATION": {
                    "p": 1.0,
                    "r": 0.9090909090909091,
                    "f": 0.9523809523809523,
                },
                "ORGANIZATION": {
                    "p": 0.5,
                    "r": 0.6666666666666666,
                    "f": 0.5714285714285715,
                },
            },
        },
        "wiki_and_books": {
            "ents_p": 0.5254237288135594,
            "ents_r": 0.7948717948717948,
            "ents_f": 0.6326530612244898,
            "ents_per_type": {
                "LOCATION": {
                    "p": 0.5806451612903226,
                    "r": 0.9,
                    "f": 0.7058823529411764,
                },
                "ORGANIZATION": {
                    "p": 0.3076923076923077,
                    "r": 0.5714285714285714,
                    "f": 0.4,
                },
                "PERSON": {"p": 0.6, "r": 0.75, "f": 0.6666666666666665},
            },
        },
        "legal": {
            "ents_p": 0.5510204081632653,
            "ents_r": 0.6923076923076923,
            "ents_f": 0.6136363636363635,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.64,
                    "r": 0.6153846153846154,
                    "f": 0.6274509803921569,
                },
                "PERSON": {"p": 0.5, "r": 0.9047619047619048, "f": 0.6440677966101696},
                "LOCATION": {"p": 0.3, "r": 0.6, "f": 0.4},
            },
        },
        "dannet": {
            "ents_p": 0.5,
            "ents_r": 1.0,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "LOCATION": {"p": 0.5, "r": 1.0, "f": 0.6666666666666666}
            },
        },
        "conversation": {
            "ents_p": 0.8235294117647058,
            "ents_r": 0.5714285714285714,
            "ents_f": 0.6746987951807228,
            "ents_per_type": {
                "LOCATION": {"p": 1.0, "r": 1.0, "f": 1.0},
                "PERSON": {
                    "p": 0.625,
                    "r": 0.2631578947368421,
                    "f": 0.37037037037037035,
                },
                "ORGANIZATION": {
                    "p": 0.5714285714285714,
                    "r": 0.36363636363636365,
                    "f": 0.4444444444444444,
                },
            },
        },
        "social_media": {
            "ents_p": 0.5384615384615384,
            "ents_r": 0.4666666666666667,
            "ents_f": 0.5,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.6666666666666666,
                    "r": 0.3333333333333333,
                    "f": 0.4444444444444444,
                },
                "PERSON": {"p": 0.5, "r": 0.5714285714285714, "f": 0.5333333333333333},
                "LOCATION": {"p": 0.5, "r": 0.5, "f": 0.5},
            },
        },
        "all_domains": {
            "ents_p": 0.6189069423929099,
            "ents_r": 0.6736334405144695,
            "ents_f": 0.645111624326405,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.6341463414634146,
                    "r": 0.7514450867052023,
                    "f": 0.6878306878306879,
                },
                "LOCATION": {
                    "p": 0.7155172413793104,
                    "r": 0.8097560975609757,
                    "f": 0.759725400457666,
                },
                "ORGANIZATION": {
                    "p": 0.5125,
                    "r": 0.5040983606557377,
                    "f": 0.5082644628099172,
                },
            },
        },
    },
    "da_core_news_sm": {
        "web": {
            "ents_p": 0.2962085308056872,
            "ents_r": 0.3056234718826406,
            "ents_f": 0.3008423586040915,
            "ents_per_type": {
                "PERSON": {"p": 0.28654970760233917, "r": 0.5, "f": 0.3643122676579926},
                "LOCATION": {
                    "p": 0.4473684210526316,
                    "r": 0.3493150684931507,
                    "f": 0.3923076923076923,
                },
                "ORGANIZATION": {
                    "p": 0.18248175182481752,
                    "r": 0.15151515151515152,
                    "f": 0.16556291390728478,
                },
            },
        },
        "news": {
            "ents_p": 0.8421052631578947,
            "ents_r": 0.6666666666666666,
            "ents_f": 0.744186046511628,
            "ents_per_type": {
                "PERSON": {"p": 0.6666666666666666, "r": 0.5, "f": 0.5714285714285715},
                "LOCATION": {"p": 1.0, "r": 0.8181818181818182, "f": 0.9},
                "ORGANIZATION": {"p": 0.75, "r": 0.6, "f": 0.6666666666666665},
            },
        },
        "wiki_and_books": {
            "ents_p": 0.23863636363636365,
            "ents_r": 0.44680851063829785,
            "ents_f": 0.3111111111111111,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.16666666666666666,
                    "r": 0.3333333333333333,
                    "f": 0.2222222222222222,
                },
                "LOCATION": {
                    "p": 0.35555555555555557,
                    "r": 0.7619047619047619,
                    "f": 0.48484848484848486,
                },
                "PERSON": {
                    "p": 0.08,
                    "r": 0.11764705882352941,
                    "f": 0.09523809523809526,
                },
            },
        },
        "legal": {
            "ents_p": 0.5384615384615384,
            "ents_r": 0.40229885057471265,
            "ents_f": 0.46052631578947373,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.6296296296296297,
                    "r": 0.32075471698113206,
                    "f": 0.425,
                },
                "PERSON": {
                    "p": 0.5769230769230769,
                    "r": 0.5172413793103449,
                    "f": 0.5454545454545454,
                },
                "LOCATION": {"p": 0.25, "r": 0.6, "f": 0.35294117647058826},
            },
        },
        "dannet": {
            "ents_p": 0.5,
            "ents_r": 1.0,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "LOCATION": {"p": 0.5, "r": 1.0, "f": 0.6666666666666666}
            },
        },
        "conversation": {
            "ents_p": 0.6666666666666666,
            "ents_r": 0.4489795918367347,
            "ents_f": 0.5365853658536586,
            "ents_per_type": {
                "LOCATION": {
                    "p": 0.8571428571428571,
                    "r": 0.9473684210526315,
                    "f": 0.9,
                },
                "ORGANIZATION": {
                    "p": 0.25,
                    "r": 0.09090909090909091,
                    "f": 0.13333333333333333,
                },
                "PERSON": {
                    "p": 0.375,
                    "r": 0.15789473684210525,
                    "f": 0.22222222222222218,
                },
            },
        },
        "social_media": {
            "ents_p": 0.1875,
            "ents_r": 0.1875,
            "ents_f": 0.1875,
            "ents_per_type": {
                "ORGANIZATION": {"p": 0.0, "r": 0.0, "f": 0.0},
                "LOCATION": {"p": 0.0, "r": 0.0, "f": 0.0},
                "PERSON": {"p": 0.5, "r": 0.375, "f": 0.42857142857142855},
            },
        },
        "all_domains": {
            "ents_p": 0.34307692307692306,
            "ents_r": 0.34520123839009287,
            "ents_f": 0.3441358024691358,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.3114754098360656,
                    "r": 0.39790575916230364,
                    "f": 0.34942528735632183,
                },
                "LOCATION": {
                    "p": 0.46445497630331756,
                    "r": 0.47572815533980584,
                    "f": 0.47002398081534774,
                },
                "ORGANIZATION": {
                    "p": 0.2512820512820513,
                    "r": 0.19678714859437751,
                    "f": 0.22072072072072071,
                },
            },
        },
    },
    "da_core_news_md": {
        "web": {
            "ents_p": 0.5215189873417722,
            "ents_r": 0.5036674816625917,
            "ents_f": 0.5124378109452736,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.5847457627118644,
                    "r": 0.7040816326530612,
                    "f": 0.638888888888889,
                },
                "ORGANIZATION": {"p": 0.325, "r": 0.3151515151515151, "f": 0.32},
                "LOCATION": {
                    "p": 0.7264957264957265,
                    "r": 0.5821917808219178,
                    "f": 0.6463878326996197,
                },
            },
        },
        "news": {
            "ents_p": 0.9444444444444444,
            "ents_r": 0.7083333333333334,
            "ents_f": 0.8095238095238096,
            "ents_per_type": {
                "PERSON": {"p": 1.0, "r": 0.75, "f": 0.8571428571428571},
                "LOCATION": {
                    "p": 0.9090909090909091,
                    "r": 0.9090909090909091,
                    "f": 0.9090909090909091,
                },
                "ORGANIZATION": {"p": 1.0, "r": 0.2, "f": 0.33333333333333337},
            },
        },
        "wiki_and_books": {
            "ents_p": 0.3466666666666667,
            "ents_r": 0.5531914893617021,
            "ents_f": 0.4262295081967213,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.13636363636363635,
                    "r": 0.3333333333333333,
                    "f": 0.1935483870967742,
                },
                "LOCATION": {
                    "p": 0.47058823529411764,
                    "r": 0.7619047619047619,
                    "f": 0.5818181818181817,
                },
                "PERSON": {
                    "p": 0.3684210526315789,
                    "r": 0.4117647058823529,
                    "f": 0.3888888888888889,
                },
            },
        },
        "legal": {
            "ents_p": 0.5476190476190477,
            "ents_r": 0.5287356321839081,
            "ents_f": 0.5380116959064327,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.6785714285714286,
                    "r": 0.6551724137931034,
                    "f": 0.6666666666666666,
                },
                "ORGANIZATION": {
                    "p": 0.5,
                    "r": 0.4339622641509434,
                    "f": 0.46464646464646464,
                },
                "LOCATION": {"p": 0.4, "r": 0.8, "f": 0.5333333333333333},
            },
        },
        "dannet": {
            "ents_p": 0.5,
            "ents_r": 1.0,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "LOCATION": {"p": 0.5, "r": 1.0, "f": 0.6666666666666666}
            },
        },
        "conversation": {
            "ents_p": 0.7575757575757576,
            "ents_r": 0.5102040816326531,
            "ents_f": 0.6097560975609756,
            "ents_per_type": {
                "LOCATION": {
                    "p": 0.8571428571428571,
                    "r": 0.9473684210526315,
                    "f": 0.9,
                },
                "PERSON": {
                    "p": 0.625,
                    "r": 0.2631578947368421,
                    "f": 0.37037037037037035,
                },
                "ORGANIZATION": {
                    "p": 0.5,
                    "r": 0.18181818181818182,
                    "f": 0.26666666666666666,
                },
            },
        },
        "social_media": {
            "ents_p": 0.5,
            "ents_r": 0.4375,
            "ents_f": 0.4666666666666667,
            "ents_per_type": {
                "ORGANIZATION": {"p": 0.5, "r": 0.3333333333333333, "f": 0.4},
                "PERSON": {"p": 0.625, "r": 0.625, "f": 0.625},
                "LOCATION": {"p": 0.0, "r": 0.0, "f": 0.0},
            },
        },
        "all_domains": {
            "ents_p": 0.529505582137161,
            "ents_r": 0.5139318885448917,
            "ents_f": 0.5216025137470541,
            "ents_per_type": {
                "PERSON": {"p": 0.6, "r": 0.5968586387434555, "f": 0.5984251968503937},
                "ORGANIZATION": {
                    "p": 0.3487394957983193,
                    "r": 0.3333333333333333,
                    "f": 0.34086242299794656,
                },
                "LOCATION": {
                    "p": 0.678391959798995,
                    "r": 0.6553398058252428,
                    "f": 0.6666666666666667,
                },
            },
        },
    },
    "da_core_news_lg": {
        "web": {
            "ents_p": 0.5830985915492958,
            "ents_r": 0.5061124694376528,
            "ents_f": 0.5418848167539267,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.6403508771929824,
                    "r": 0.7448979591836735,
                    "f": 0.6886792452830189,
                },
                "LOCATION": {
                    "p": 0.7027027027027027,
                    "r": 0.5342465753424658,
                    "f": 0.6070038910505837,
                },
                "ORGANIZATION": {
                    "p": 0.4307692307692308,
                    "r": 0.3393939393939394,
                    "f": 0.3796610169491526,
                },
            },
        },
        "news": {
            "ents_p": 0.7142857142857143,
            "ents_r": 0.625,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "PERSON": {"p": 0.5714285714285714, "r": 0.5, "f": 0.5333333333333333},
                "ORGANIZATION": {"p": 0.3333333333333333, "r": 0.2, "f": 0.25},
                "LOCATION": {
                    "p": 0.9090909090909091,
                    "r": 0.9090909090909091,
                    "f": 0.9090909090909091,
                },
            },
        },
        "wiki_and_books": {
            "ents_p": 0.3611111111111111,
            "ents_r": 0.5531914893617021,
            "ents_f": 0.43697478991596644,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.16666666666666666,
                    "r": 0.2222222222222222,
                    "f": 0.1904761904761905,
                },
                "PERSON": {
                    "p": 0.21212121212121213,
                    "r": 0.4117647058823529,
                    "f": 0.27999999999999997,
                },
                "LOCATION": {
                    "p": 0.6296296296296297,
                    "r": 0.8095238095238095,
                    "f": 0.7083333333333334,
                },
            },
        },
        "legal": {
            "ents_p": 0.6521739130434783,
            "ents_r": 0.5172413793103449,
            "ents_f": 0.576923076923077,
            "ents_per_type": {
                "PERSON": {"p": 0.6, "r": 0.6206896551724138, "f": 0.6101694915254238},
                "ORGANIZATION": {
                    "p": 0.7096774193548387,
                    "r": 0.41509433962264153,
                    "f": 0.5238095238095237,
                },
                "LOCATION": {"p": 0.625, "r": 1.0, "f": 0.7692307692307693},
            },
        },
        "dannet": {
            "ents_p": 0.5,
            "ents_r": 1.0,
            "ents_f": 0.6666666666666666,
            "ents_per_type": {
                "LOCATION": {"p": 0.5, "r": 1.0, "f": 0.6666666666666666}
            },
        },
        "conversation": {
            "ents_p": 0.675,
            "ents_r": 0.5510204081632653,
            "ents_f": 0.6067415730337078,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.4166666666666667,
                    "r": 0.2631578947368421,
                    "f": 0.3225806451612903,
                },
                "LOCATION": {
                    "p": 0.9473684210526315,
                    "r": 0.9473684210526315,
                    "f": 0.9473684210526315,
                },
                "ORGANIZATION": {
                    "p": 0.4444444444444444,
                    "r": 0.36363636363636365,
                    "f": 0.39999999999999997,
                },
            },
        },
        "social_media": {
            "ents_p": 0.47058823529411764,
            "ents_r": 0.5,
            "ents_f": 0.48484848484848486,
            "ents_per_type": {
                "ORGANIZATION": {
                    "p": 0.3333333333333333,
                    "r": 0.16666666666666666,
                    "f": 0.2222222222222222,
                },
                "LOCATION": {
                    "p": 0.14285714285714285,
                    "r": 0.5,
                    "f": 0.22222222222222224,
                },
                "PERSON": {"p": 0.8571428571428571, "r": 0.75, "f": 0.7999999999999999},
            },
        },
        "all_domains": {
            "ents_p": 0.5689655172413793,
            "ents_r": 0.5108359133126935,
            "ents_f": 0.5383360522022839,
            "ents_per_type": {
                "PERSON": {
                    "p": 0.5588235294117647,
                    "r": 0.5968586387434555,
                    "f": 0.5772151898734178,
                },
                "LOCATION": {
                    "p": 0.6989247311827957,
                    "r": 0.6310679611650486,
                    "f": 0.663265306122449,
                },
                "ORGANIZATION": {
                    "p": 0.45263157894736844,
                    "r": 0.3453815261044177,
                    "f": 0.39179954441913445,
                },
            },
        },
    },
}

import pandas as pd

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
domain_f1.rename(columns={"all_domains": "across_domains"}, inplace=True).sort_values(
    by=["Model"]
)
domain_f1

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
tag_f1
tag_f1_wide
