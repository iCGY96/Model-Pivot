'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''

from setuptools import find_packages
from setuptools import setup

requirements = ["torch"]

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    install_requires=install_requires,
    name="Model-Pivot",
    version="0.0.1",
    author="PKU MLG, PCL, and other contributors",
    author_email="gychen@pku.edu.cn",
    description="A model conversion and visualization tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.openi.org.cn/OpenI/Model-Pivot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)