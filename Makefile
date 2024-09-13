VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit
GRADIO = $(VENV)/bin/gradio

# Load .env file
include .env
export

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

chatopenai: $(VENV)/bin/activate
	$(STREAMLIT) run chatopenai.py

chat: $(VENV)/bin/activate
	$(STREAMLIT) run chat.py --server.port 8502

chatpdf: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdf.py

chatpdfemb: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdfemb.py

chatgradio: $(VENV)/bin/activate
	$(GRADIO) chatgradio.py

search: $(VENV)/bin/activate
	$(STREAMLIT) run chatsearch.py --server.port 8503

discussion: $(VENV)/bin/activate
	$(STREAMLIT) run discussion.py --server.port 9093

clean:
	rm -rf __pycache__
	rm -rf $(VENV)