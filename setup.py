import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xmbwsghmc", # Replace with your own username
    version="0.0.1",
    author="Xiangwen Mo, Bingruo Wu",
    author_email="xiangwen.mo@duke.edu, bingruo.wu@duke.edu",
    description="Stochastic Gradient Hamilton Monte Carlo",
    url="https://github.com/xiangwenmo/SGHMC-final-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)