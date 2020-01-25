import setuptools

from beampy import __version__ as beampy_version
# https://packaging.python.org/tutorials/packaging-projects/

# use this command in Beampy folder to create the pypi folder:
# python setup.py sdist bdist_wheel

# Then use this command to upload to pypi:
# twine upload dist/*

with open("README.rst", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='beampy',
    version=beampy_version,
    author="Jonathan Peltier and Marcel Soubkovsky",
    author_email="jonathanp57@outlook.fr",
    license="MIT License",
    description="""Beampy is a python package - with an user interface -
    allowing to propagate beams in differents guides using the
    Beam Propagation Method (BPM)""",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url="https://github.com/Python-simulation/Beampy",
    project_urls={
        'Documentation': 'https://beampy.readthedocs.io',
        'Source Code': 'https://github.com/Python-simulation/Beampy'
    },
    packages=setuptools.find_packages(),
	include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
            "numpy>=1.16",
            "PyQt5>=5.13",
            "matplotlib>=3.1",
        ],
)
