import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="object_detection",
    version="0.1.6",
    author="Muted Permit",
    author_email="example@example.com",
    description="A pip installable package for object detection based on tensorflow/models/object_detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MutedPermit/object_detection",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
