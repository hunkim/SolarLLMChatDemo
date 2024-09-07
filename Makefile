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

search: $(VENV)/bin/activate
	$(STREAMLIT) run chatyousearch.py

discussion: $(VENV)/bin/activate
	$(STREAMLIT) run discussion.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)