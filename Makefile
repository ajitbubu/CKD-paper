# CKD JMIR PHS Submission - end-to-end pipeline
# Run `make all` from repo root for a fresh build from raw downloads to manuscript.
#
# Targets are dependency-ordered.  Each step writes its outputs to disk so the
# next step can pick up where the prior left off.

PY := python3
ROOT := $(shell pwd)

.PHONY: all data train figures evaluate paper clean help

help:
	@echo "Targets:"
	@echo "  data       — fetch ADI, CDC PLACES, ZCTA crosswalk, NHANES kidney panel"
	@echo "  train      — train ecological model + NHANES classifier"
	@echo "  evaluate   — full evaluation, calibration, decision curves, subgroups"
	@echo "  figures    — regenerate ecological + NHANES figure set"
	@echo "  paper      — rewrite the JMIR docx from live metric files"
	@echo "  all        — data → train → evaluate → figures → paper"
	@echo "  clean      — remove generated artefacts (keeps raw data)"

# ----------------------------------------------------------------- data
data: data/cdc_places/places_ckd_2022_tract.csv \
      data/census/zcta_tract_crosswalk_2020.csv \
      data/nhanes/nhanes_kidney_panel_2017_2023.csv

data/cdc_places/places_ckd_2022_tract.csv: src/data_processing/fetch_cdc_places.py
	$(PY) $<

data/census/zcta_tract_crosswalk_2020.csv: src/data_processing/fetch_zcta_crosswalk.py
	$(PY) $<

data/nhanes/nhanes_kidney_panel_2017_2023.csv: src/data_processing/fetch_nhanes.py
	$(PY) $<

# ----------------------------------------------------------------- train
train: models/ckd_ecological_model.pkl models/ckd_nhanes_classifier.pkl

models/ckd_ecological_model.pkl: src/train_ecological_model.py \
                                 data/adi_2020_national_blockgroup.csv \
                                 data/cdc_places/places_ckd_2022_tract.csv \
                                 data/census/state_to_region.csv
	$(PY) $<

models/ckd_nhanes_classifier.pkl: src/train_nhanes_model.py \
                                  data/nhanes/nhanes_kidney_panel_2017_2023.csv
	$(PY) $<

# ----------------------------------------------------------------- evaluate
evaluate: results/metrics/nhanes_performance.json

results/metrics/nhanes_performance.json: src/evaluate_nhanes_model.py \
                                         models/ckd_nhanes_classifier.pkl
	$(PY) $<

# ----------------------------------------------------------------- figures
figures: train evaluate
	$(PY) src/generate_ecological_figures.py

# ----------------------------------------------------------------- paper
paper: white_paper/CKD_Paper_JMIR_PHS_v3.docx

white_paper/CKD_Paper_JMIR_PHS_v3.docx: src/rewrite_white_paper.py figures
	$(PY) $<

# ----------------------------------------------------------------- top-level
all: data train evaluate figures paper
	@echo "Pipeline complete.  Manuscript: white_paper/CKD_Paper_JMIR_PHS_v3.docx"

clean:
	rm -rf results/figures/*.png
	rm -f results/metrics/*.json results/metrics/*.csv
	rm -f results/training_metrics_ecological.json
	rm -f results/tract_ckd_predictions.csv
	rm -f results/nhanes_oof_predictions.csv
	rm -f white_paper/CKD_Paper_JMIR_PHS_v3.docx
	@echo "Clean.  (Raw data and trained models preserved.)"
