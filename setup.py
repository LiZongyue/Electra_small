from setuptools import setup, find_packages

setup(
   name='Electra',
   version='3.1.1',
   description='Train a small Electra from scratch and fine tune it with GLUE(SST-2) and with ImDb data. After '
               'getting the result, compare it with Bert.',
   author='Yifu Chen and Zongyue Li',
   author_email='zongyue.li@outlook.com',
   package_dir={"": "src"},
   packages=find_packages("src"), install_requires=['transformers', 'scikit-learn', 'torch', 'numpy', 'pandas',
                                                    'termcolor', 'tqdm', 'matplotlib']
)
