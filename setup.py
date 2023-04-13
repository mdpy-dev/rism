from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rism",
    version="0.1.0",
    author="Zhenyu Wei",
    author_email="zhenyuwei99@gmail.com",
    description="RISM is a python solver for RISM and OZ model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source Code": "https://github.com/mdpy-dev/rism",
    },
    packages=find_packages(),
    package_data={
        "mdpy": [
            "test/data/*",
        ]
    },
    # setup_requires = ['pytest-runner'],
    tests_require=["pytest"],
    install_requires=[
        "numpy >= 1.20.0",
        "scipy >= 1.7.0",
        "matplotlib >= 3.0.0",
        "pytest >= 6.2.0",
        "cupy>=10.2.0",
    ],
    python_requires=">=3.9",
)
