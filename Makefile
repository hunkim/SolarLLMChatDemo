# Define the two virtual environments
VENV = .venv
BROWSER_VENV = .venv-browser
PYTHON = $(VENV)/bin/python3
BROWSER_PYTHON = $(BROWSER_VENV)/bin/python3
PIP = $(VENV)/bin/pip3
BROWSER_PIP = $(BROWSER_VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit
BROWSER_STREAMLIT = $(BROWSER_VENV)/bin/streamlit
GRADIO = $(VENV)/bin/gradio

# Basic venv without browser dependencies
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Browser-enabled venv with playwright
$(BROWSER_VENV)/bin/activate: requirements.txt
	python3 -m venv $(BROWSER_VENV)
	$(BROWSER_PIP) install -r requirements.txt
	$(BROWSER_PIP) install playwright
	$(BROWSER_PYTHON) -m playwright install

chatopenai: $(VENV)/bin/activate
	$(STREAMLIT) run chatopenai.py

coldmail: $(VENV)/bin/activate
	$(STREAMLIT) run coldmail.py

chat: $(VENV)/bin/activate
	$(STREAMLIT) run chat.py 

chatpdf: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdf.py

longimg: $(VENV)/bin/activate
	$(STREAMLIT) run longimg.py

chatpdfemb: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdfemb.py

gemini: $(VENV)/bin/activate
	$(STREAMLIT) run gemini.py

chatgradio: $(VENV)/bin/activate
	$(GRADIO) chatgradio.py

docv: $(VENV)/bin/activate
	$(STREAMLIT) run docv.py

search: $(VENV)/bin/activate
	$(STREAMLIT) run chatsearch.py 

reasoning: $(VENV)/bin/activate
	$(STREAMLIT) run reasoning.py 

discussion: $(VENV)/bin/activate
	$(STREAMLIT) run discussion.py --server.port 9093

llama: $(VENV)/bin/activate
	$(STREAMLIT) run llama.py

voice: $(VENV)/bin/activate
	$(STREAMLIT) run voice.py

hw: $(VENV)/bin/activate
	$(STREAMLIT) run hw.py

util: $(VENV)/bin/activate
	$(PYTHON) solar_util.py

podcast: $(VENV)/bin/activate
	$(STREAMLIT) run podcast.py

biz_help: $(VENV)/bin/activate
	$(STREAMLIT) run biz_help.py

info_fill: $(BROWSER_VENV)/bin/activate
	$(BROWSER_STREAMLIT) run info_fill.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf $(BROWSER_VENV)