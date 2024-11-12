from setuptools import setup

setup(
    name='patra_model_card',
    version='0.1.1',
    packages=['tests', 'patra_model_card'],
    package_data={'patra_model_card': ['schema/schema.json']},
    include_package_data=True,
    url='https://github.com/Data-to-Insight-Center/patra-toolkit.git',
    license='BSD-3-Clause',
    author='Data to Insight Center',
    author_email='d2i@iu.edu',
    description='Patra Model Card Toolkit',
    install_requires=[
        'attrs~=23.1.0',
        'jsonschema~=4.18.6',
        'fairlearn~=0.11.0',
        'shap~=0.46.0',
        'pandas>=2.0.0',
        'numpy>2.0.0',
        'pyrsistent~=0.19.3',
        'requests>2.32.2',
    ]
)
