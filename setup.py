from setuptools import setup
# https://python-packaging.readthedocs.io/en/latest/minimal.html
setup(name='cropping_lib',
      version='0.0.1',
      description='Library to run some cropping things',
      url='TBD',
      author='SR Steinkamp',
      author_email='tbd',
      license='tbd',
      packages=['cropping_lib'],
      scripts=['scripts/investigate_data.py',
               'scripts/prepare_data.py',
               'scripts/evaluate_model.py',
               'scripts/train_model.py',
               'scripts/predict_cli.py'],
      zip_safe=False)