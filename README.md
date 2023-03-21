# First acquire data/dansk.spacy 

# Split DANSK into train/dev/test
python src/fetch_assets.py

# Get DANSK full to jsonl also
python src/load_docbin_as_jsonl.py data/DANSK/full.spacy blank:da --ner > data/DANSK/full.jsonl

# Get descriptive stats on partitions
python src/descriptive_stats_partitions.py

# Get descriptive stats on domains within partitions
python src/descriptive_stats_domains.py

# Get test_domain.spacy into "old annotation format" (only PER, LOC and ORG)
python src/dansk_test_to_old_annotation_format.py

# Test new DaCy models on test.spacy
- F1-score, recall, precision within:
        - Domains (across ents)
        - Ent_types (across domains)
        - Domains + Ents (all combinations)
python src/test_fine_grained_models.py

# Get interrater reliability stats within annotators
python src/inter_annotators_matrix.py
inter_annotators_plots.rmd

# Get interrater reliability stats DANSK vs. annotators
python src/inter_annotators_vs_DANSK_matrix.py
inter_annotators_vs_DANSK_plots.rmd

# Test other models on test.spacy (maybe including saattrupdan/nbailab-base-ner-scandi)
    - F1-score, recall, precision within:
        - Domains (across ents)
        - Ent_types (across domains)
python src/test_models.py

# Get interrater reliability stats DANSK vs da_dacy_size_ner_fine_grained (x3)
python src/inter_model_vs_DANSK_matrix.py







# Gotten to here






# Generate plots from data
src/plots_ggpot.Rmd
inter_model_vs_DANSK_plots.rmd

