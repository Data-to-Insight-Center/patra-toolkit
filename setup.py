from setuptools import setup

setup(
    name='patra_model_card',
    version='0.2',
    packages=['tests', 'patra_model_card'],
    package_data={'patra_model_card': ['schema/schema.json']},
    include_package_data=True,
    url='https://github.com/Data-to-Insight-Center/patra-toolkit.git',
    license='BSD-3-Clause',
    author='Data to Insight Center',
    author_email='d2i@indiana.edu',
    description='Patra Model Card Toolkit',
    install_requires=[
        'attrs~=23.1.0',
        'jsonschema~=4.18.6',
        'fairlearn~=0.9.0',
        'scikit-learn~=1.3.1'
    ]
)
