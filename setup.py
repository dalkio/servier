# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['servier', 'servier.api', 'servier.model', 'servier.utils']

package_data = \
{'': ['*']}

install_requires = \
['flask>=3.0.3,<4.0.0',
 'gunicorn>=22.0.0,<23.0.0',
 'loguru>=0.7.2,<0.8.0',
 'numpy>=1.26.4,<2.0.0',
 'omegaconf==2.3.0',
 'pandas>=2.2.2,<3.0.0',
 'pydantic>=2.7.1,<3.0.0',
 'rdkit==2023.9.5',
 'scikit-learn==1.4.2',
 'torch==2.2.2',
 'uvicorn>=0.30.0,<0.31.0']

setup_kwargs = {
    'name': 'servier',
    'version': '0.1.0',
    'description': 'Simple molecule classification project',
    'long_description': '# servier\n\n- `poetry install --with dev --no-root`\n- `poetry shell`\n- `poetry2setup > setup.py`\n- `python setup.py install`\n- `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d \'{"input": "Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C"}\'`\n\n- `docker build --platform=linux/amd64 -t servier:latest .`\n- `docker run -p 8000:8000 -v $(pwd)/models:/app/models servier`\n',
    'author': 'Côme Arvis',
    'author_email': 'come.arvis@neuronest.net',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)

