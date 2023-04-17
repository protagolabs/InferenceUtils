import codecs
import os
import re

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "inferenceUtils/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

setup(
    name="inferenceUtils",
    version=version_string,
    packages=["inferenceUtils"],
    install_requires=[
        "async-timeout==4.0.2",
        "boto3==1.24.28",
        "botocore==1.27.59",
        "brotlipy==0.7.0",
        "certifi==2022.12.7",
        "cffi==1.15.1",
        "cryptography==39.0.1",
        "idna==3.4",
        "jmespath==0.10.0",
        "pip==23.0.1",
        "pycparser==2.21",
        "pyOpenSSL==23.0.0",
        "PySocks==1.7.1",
        "python-dateutil==2.8.2",
        "redis==4.5.3",
        "s3transfer==0.6.0",
        "setuptools==65.6.3",
        "six==1.16.0",
        "urllib3==1.26.14",
        "wheel==0.38.4",
        "requests==2.28.2"
    ],
    description="package for inference at net-mind platform"
)
