from setuptools import find_packages, setup

setup(
    name="clip_saver",
    version="0.9",
    packages=find_packages(),
    install_requires=[
        "lapx",
        "supervision",
        "ultralytics",
    ],
)
