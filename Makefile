












PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
SCRIPT ?= make_ref.py
STREAMLIT ?= .venv/bin/streamlit
APP       ?= app.py
HOST      ?= 127.0.0.1
PORT      ?= 8501

OUTDIR := results         
ARTIFACT := artifacts    
PAIRS := \
  test_images/closedoor.png:test_images/opendoor.png \
  test_images/closedoor.png:test_images/closedoor_different_colors.png \
  test_images/easy.png:test_images/easy_wrong.png

.PHONY: help venv deps clean clean-artifacts clean-results test-all run test1 test2 app app-dev app-open streamlit-cache-clear

help:
	@echo "make deps           
	@echo "make test-all       
	@echo "make run A=img1 B=img2  
	@echo "make clean  

venv:
	@test -d .venv || python3 -m venv .venv

deps: venv
	$(PIP) install --upgrade pip
	$(PIP) install opencv-python scikit-image numpy streamlit streamlit-autorefresh







run: venv
	@if [ -z "$(A)" ] || [ -z "$(B)" ]; then \
		echo "Usage: make run A=<imgA> B=<imgB>"; exit 1; \
	fi
	@name="$${A%.*}_vs_$${B%.*}"; \
	echo "==> $$name"; \
	rm -rf $(OUTDIR); mkdir -p $(OUTDIR); \
	printf '%s\n' n | $(PYTHON) $(SCRIPT) "$(A)" "$(B)"; \
	mkdir -p "$(ARTIFACT)/$$name"; \
	if ls $(OUTDIR)/* >/dev/null 2>&1; then mv $(OUTDIR)/* "$(ARTIFACT)/$$name/"; fi; \
	echo "saved to $(ARTIFACT)/$$name"



test-all: venv
	@rm -rf "$(ARTIFACT)"; mkdir -p "$(ARTIFACT)"
	@for P in $(PAIRS); do \
		A="$${P%%:*}"; B="$${P##*:}"; \
		name="$${A%.*}_vs_$${B%.*}"; \
		echo "==> $$name"; \
		rm -rf $(OUTDIR); mkdir -p $(OUTDIR); \
		printf '%s\n' n | $(PYTHON) $(SCRIPT) "$$A" "$$B"; \
		mkdir -p "$(ARTIFACT)/$$name"; \
		if ls $(OUTDIR)/* >/dev/null 2>&1; then mv $(OUTDIR)/* "$(ARTIFACT)/$$name/"; fi; \
	done
	@echo "All artifacts -> $(ARTIFACT)/<case>/"


test1:
	$(MAKE) run A=test_images/closedoor.png B=test_images/opendoor.png
test2:
	$(MAKE) run A=test_images/closedoor.png B=test_images/closedoor_different_colors.png
test3:
	$(MAKE) run A=test_images/easy.png B=test_images/easy_wrong.png


test4:
	echo "拡大画像"
	$(MAKE) run A=test_images/IMG_4726.PNG B=test_images/IMG_4728.PNG
	
test5:
	echo "拡大画像逆ver"
	$(MAKE) run A=test_images/IMG_4728.PNG B=test_images/IMG_4726.PNG

clean-results:
	@rm -rf "$(OUTDIR)"
clean-artifacts:
	@rm -rf "$(ARTIFACT)"
clean: clean-results clean-artifacts

app: venv deps
	$(STREAMLIT) run $(APP) \
	  --server.address=$(HOST) \
	  --server.port=$(PORT)

app-dev: venv deps
	BROWSER=none $(STREAMLIT) run $(APP) \
	  --server.address=$(HOST) \
	  --server.port=$(PORT)




app-open:
	@which open >/dev/null 2>&1 && open "http://$(HOST):$(PORT)" || true

streamlit-cache-clear:
	@echo "Clearing Streamlit cache..."
	@rm -rf ~/.cache/streamlit
	@echo "Done."