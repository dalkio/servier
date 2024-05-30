from setuptools import setup, find_packages

setup(
    name='servier',
    version='0.1.0',
    description='Simple molecule classification project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Côme Arvis',
    author_email='come.arvis@neuronest.net',
    packages=find_packages(include=['servier', 'servier.*']),
    include_package_data=True,
    package_data={
        '': ['servier/config.yaml'],
    },
    install_requires=[
        'flask>=3.0.3,<4.0.0',
        'gunicorn>=22.0.0,<23.0.0',
        'loguru>=0.7.2,<0.8.0',
        'numpy>=1.26.4,<2.0.0',
        'omegaconf==2.3.0',
        'pandas>=2.2.2,<3.0.0',
        'pydantic>=2.7.1,<3.0.0',
        'rdkit==2023.9.5',
        'scikit-learn==1.4.2',
        'torch==2.2.2',
        'transformers>=4.41.1,<5.0.0'
    ],
    python_requires='>=3.9,<4.0',
    entry_points={
        'console_scripts': [
            'servier=servier.main:main',
        ],
    },
)
