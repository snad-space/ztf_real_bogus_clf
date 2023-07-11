from setuptools import setup, find_packages



with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
		'numpy==1.23.5',
		'onnxruntime==1.15.1',
		'pandas==2.0.0',
		'Requests==2.31.0',
		'scikit_learn==1.2.2',
		'setuptools==67.4.0',
		'skl2onnx==1.14.1',
		'tqdm==4.65.0',
		]

setup(
    name="RBclf",
    version="0.0.1",
    author='Timofey Semenikhin'
    author_email="ofmafowo@gmail.com",
    description="Real-bogus classification for ZTF objects.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/snad-space/ztf_real_bogus_clf",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)
