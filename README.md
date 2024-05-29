# servier

- `poetry install --with dev --no-root`
- `poetry shell`
- `poetry2setup > setup.py`
- `python setup.py install`
- `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"input": "Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C", "model_name": "chem_berta"}'`

- `docker build --platform=linux/amd64 -t servier:latest .`
- `docker run -p 8000:8000 -v $(pwd)/models:/app/models servier`
