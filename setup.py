from setuptools import find_packages, setup

setup(
    name="clip_saver",
    version="0.7.2",
    packages=find_packages(),
    install_requires=[
        "lapx",
        "numpy",
        "pydantic",
        "supervision",
        "ultralytics",
    ],
)
