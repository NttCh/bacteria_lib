from setuptools import setup, find_packages

setup(
    name="bacteria_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "torch",
        "torchvision",
        "albumentations",
        "pytorch-lightning",
        "omegaconf",
        "scikit-learn",
        "optuna",
        "matplotlib",
        "seaborn",
    ],
    author="Your Name",
    description="A library for bacteria detection and classification using deep learning.",
    long_description="A longer description for your library, or read from a README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bacteria_lib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
