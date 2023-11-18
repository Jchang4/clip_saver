from setuptools import find_packages, setup

setup(
    name="clip_saver",
    version="0.3",
    packages=find_packages(),
    install_requires=["ultralytics", "supervision", "numpy", "pydantic"],
)
