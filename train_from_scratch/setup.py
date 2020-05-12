from setuptools import setup, find_packages

setup(
   name='Electra',
   version='1.0',
   description='Train a smaller Electra from scratch and fine tune it with GLUE',
   author='Zongyue Li',
   author_email='zongyue.li@outlook.com',
   package_dir={"": "src"},
   packages=find_packages("src"),
)
