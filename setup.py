from setuptools import setup, find_packages

setup(
	name="info_iv",
	version="0.0.1",
	description="Instrumental variable / contrastive learning experiments",
	packages=find_packages(exclude=("tests", "docs")),
	include_package_data=True,
	install_requires=[
		"torch",
		"pytorch-lightning",
		"hydra-core",
	],
	python_requires='>=3.8',
)

