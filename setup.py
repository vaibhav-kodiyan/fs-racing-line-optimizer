from setuptools import setup, find_packages

setup(
    name="fmsim",
    version="0.1.0",
    packages=find_packages(include=['fmsim', 'fmsim.*']),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.22',
        'scipy>=1.8.0',
        'matplotlib>=3.5',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'flake8>=6.0',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Formula Student Racing Line Optimizer",
    url="https://github.com/yourusername/fs-racing-line-optimizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
