from setuptools import find_packages, setup

setup(
    name="clip_saver",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "lapx",
        "supervision",
        "ultralytics",
    ],
)
