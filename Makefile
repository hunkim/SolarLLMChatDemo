VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit
GRADIO = $(VENV)/bin/gradio

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

chatopenai: $(VENV)/bin/activate
	$(STREAMLIT) run chatopenai.py

chat: $(VENV)/bin/activate
	$(STREAMLIT) run chat.py 

chatpdf: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdf.py

chatpdfemb: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdfemb.py

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

hw: $(VENV)/bin/activate
	$(STREAMLIT) run hw.py

util: $(VENV)/bin/activate
	$(PYTHON) solar_util.py

biz_help: $(VENV)/bin/activate
	$(STREAMLIT) run biz_help.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)