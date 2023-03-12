import spacy

!pip install https://huggingface.co/chcaa/da_dacy_small_DANSK_ner/resolve/main/da_dacy_small_DANSK_ner-any-py3-none-any.whl
nlp = spacy.load("da_core_news_sm", exclude="ner")
nlp_ner = spacy.load("da_dacy_small_DANSK_ner")
nlp.add_pipe(factory_name="transformer", source=nlp_ner)
nlp.add_pipe(factory_name="ner", source=nlp_ner)

doc = nlp("Ord som Aarhus og kl. 07:30 bliver i denne tekst annoteret")

for ent in doc.ents:
    print(ent)
    print(ent.label_)

# Aarhus
# GPE
# kl. 07:30
# TIME
