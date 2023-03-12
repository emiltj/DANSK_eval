import dacy

for model in dacy.models():
    print(model)

small = dacy.load("da_dacy_small_trf-0.1.0")
print(small("Dette er en tekst om Aarhus").ents)
large = dacy.load("da_dacy_large_trf-0.1.0")
print(large("Dette er en tekst om Aarhus").ents)
