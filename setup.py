import os
import setuptools


PATH_TO_THIS_SCRIPT = os.path.abspath(os.path.dirname(__file__))


setuptools.setup(
    name="mt",
    version="0.0.1",
    author="Laurin Luttmann",
    description="master thesis project for unbiased neural ranking",
    packages=setuptools.find_packages(exclude=("mt/archive",)),
    include_package_data=True,
    python_requires='>=3.8',
    entry_points='''
        [console_scripts]
        pbk_train=mt.run.pbk_classification:main
        encoder_train=mt.run.train_encoder:main
        ranker_train=mt.run.train_ranker:main
    ''',
)

