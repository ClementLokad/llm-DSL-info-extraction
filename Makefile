# Makefile for Envision PSC Project (INF01)
# Main entry point for building deliverables and syncing

FLYER_DIR = docs/flyer
ASSETS_DIR = docs/assets
FLYER_OUT = $(ASSETS_DIR)/flyer.pdf
COMMIT_MSG = "chore: finalize psc deliverables for INF01"
DOCS_COMMIT_MSG = "docs: update documentation and assets"

.PHONY: all flyer push push-docs clean help

all: flyer clean push

help:
	@echo "Usage:"
	@echo "  make flyer      Compile the flyer PDF and move to $(ASSETS_DIR)/"
	@echo "  make push       Commit and push all changes to GitHub"
	@echo "  make push-docs  Commit and push only the docs/ directory"
	@echo "  make clean      Remove LaTeX temporary files"
	@echo "  make all        Run flyer, clean, and then push everything"

flyer:
	@echo "Compiling flyer..."
	cd $(FLYER_DIR) && pdflatex -interaction=nonstopmode flyer.tex
	mv $(FLYER_DIR)/flyer.pdf $(FLYER_OUT)
	@echo "Flyer PDF updated in $(FLYER_OUT)"

push:
	@echo "Pushing all changes to repository..."
	git add .
	git commit -m $(COMMIT_MSG)
	git push
	@echo "Successfully pushed all to GitHub."

push-docs:
	@echo "Pushing documentation changes..."
	git add docs/
	git commit -m $(DOCS_COMMIT_MSG)
	git push
	@echo "Successfully pushed docs to GitHub."

clean:
	@echo "Cleaning up debris in $(FLYER_DIR)..."
	cd $(FLYER_DIR) && rm -f *.aux *.log *.out *.toc *.nav *.snm *.fdb_latexmk *.fls *.synctex.gz
