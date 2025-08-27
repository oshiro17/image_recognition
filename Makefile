# -------- 基本設定 --------
PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
SCRIPT ?= make_ref.py

OUTDIR := results          # make_ref.py の出力先
ARTIFACT := artifacts      # ケースごとの成果物保存先

# A:B 形式で列挙（コロンで区切る）
PAIRS := \
  closedoor.png:opendoor.png \
  closedoor.png:closedoor_different_colors.png \
  easy.png:easy_wrong.png

.PHONY: help venv deps clean clean-artifacts clean-results test-all run test1 test2

help:
	@echo "make deps            # 依存インストール"
	@echo "make test-all        # 全テスト実行（artifacts/<case>/ に保存）"
	@echo "make run A=img1 B=img2  # 任意ペアを一回実行"
	@echo "make clean           # 生成物削除"

venv:
	@test -d .venv || python3 -m venv .venv

deps: venv
	$(PIP) install --upgrade pip
	$(PIP) install opencv-python scikit-image numpy

# 任意ペアを一回実行: 例) make run A=test1a.png B=test1b.png
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

# すべてのペアを順に実行
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

# お好みでショートカット
test1:
	$(MAKE) run A=closedoor.png B=opendoor.png
test2:
	$(MAKE) run A=closedoor.png B=closedoor_different_colors.png
test3:
	$(MAKE) run A=easy.png B=easy_wrong.png

clean-results:
	@rm -rf "$(OUTDIR)"
clean-artifacts:
	@rm -rf "$(ARTIFACT)"
clean: clean-results clean-artifacts