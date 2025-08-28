# -------- 基本設定 --------
PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
# 実行するスクリプトをYOLO差分検知用に変更
SCRIPT ?= run_yolo_diff.py
STREAMLIT ?= .venv/bin/streamlit
APP       ?= app.py
HOST      ?= 127.0.0.1
PORT      ?= 8501

OUTDIR := results          # スクリプトの出力先
ARTIFACT := artifacts      # ケースごとの成果物保存先

# A:B 形式で列挙（コロンで区切る）
PAIRS := \
  test_images/closedoor.png:test_images/opendoor.png \
  test_images/closedoor.png:test_images/closedoor_different_colors.png \
  test_images/easy.png:test_images/easy_wrong.png

.PHONY: help venv deps clean clean-artifacts clean-results test-all run test1 test2 app app-dev app-open streamlit-cache-clear

help:
	@echo "make deps            # 依存インストール"
	@echo "make test-all        # 全テスト実行（artifacts/<case>/ に保存）"
	@echo "make run A=img1 B=img2  # 任意ペアを一回実行"
	@echo "make clean           # 生成物削除"
	@echo "make app             # Streamlit UIを起動"

venv:
	@test -d .venv || python3 -m venv .venv

deps: venv
	$(PIP) install --upgrade pip
	$(PIP) install opencv-python scikit-image numpy streamlit streamlit-autorefresh ultralytics torch

# 任意ペアを一回実行: 例) make run A=test1a.png B=test1b.png
run: venv
	@if [ -z "$(A)" ] || [ -z "$(B)" ]; then \
		echo "Usage: make run A=<imgA> B=<imgB>"; exit 1; \
	fi
	@name="$${A%.*}_vs_$${B%.*}"; \
	@safename="$${name//\//_}"; \
	echo "==> $$safename"; \
	rm -rf $(OUTDIR); \
	$(PYTHON) $(SCRIPT) "$(A)" "$(B)"; \
	mkdir -p "$(ARTIFACT)"; \
	mv "$(OUTDIR)" "$(ARTIFACT)/$$safename"; \
	echo "saved to $(ARTIFACT)/$$safename"

# すべてのペアを順に実行
test-all: venv
	@rm -rf "$(ARTIFACT)"; mkdir -p "$(ARTIFACT)"
	@for P in $(PAIRS); do \
		A="$${P%%:*}"; B="$${P##*:}"; \
		name="$${A%.*}_vs_$${B%.*}"; \
		safename="$${name//\//_}"; \
		echo "==> $$safename"; \
		rm -rf $(OUTDIR); \
		$(PYTHON) $(SCRIPT) "$$A" "$$B"; \
		mv "$(OUTDIR)" "$(ARTIFACT)/$$safename"; \
	done
	@echo "All artifacts -> $(ARTIFACT)/<case>/"

# お好みでショートカット
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

# ---- Streamlit GUI targets ----
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
