from setuptools import find_packages, setup

setup(
    name="clip_saver",
    version="0.1",
    packages=find_packages(),
    install_requires=["ultralytics", "supervision", "numpy", "pydantic"],
)
