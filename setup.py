from setuptools import setup, find_packages

setup(
    name="ivt",
    version="0.1.0",
    description="Velocity Threshold Identification (I-VT) algorithm for eye-tracking data",
    author="Joash Ye",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "matplotlib>=3.5",
        "openpyxl>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "ivt = ivt.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
