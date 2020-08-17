import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KnowYourPlates",
    version="0.1.1",
    author="Wojciech Sikora",
    author_email="kontakt@sikorawojciech.pl",
    description="Module that allows to recognize license plates from images basing on image processing algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SikoraWojciech/KnowYourPlates",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
