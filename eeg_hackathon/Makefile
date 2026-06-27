# EEG Hackathon Kit — convenience targets
.PHONY: verify reproduce demo

verify:        ## fast synthetic smoke test (~13s, no network)
	python verify_all.py

reproduce:     ## reproduce + assert every REAL headline number (downloads; ~15-20 min)
	pip install -r requirements-real.txt
	python verify_all.py --real

demo:          ## interactive launcher menu
	python run.py
