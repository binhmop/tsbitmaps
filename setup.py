from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='tsbitmaps',
      version='0.1',
      description='Time-series Bitmap implementation for anomaly detection on time series data',
      long_description=readme(),
      url='http://github.com/binhmop/tsbitmaps',
      author='Binh Han',
      author_email='binhhan.gt@gmail.com',
      license='Apache 2.0',
      packages=['tsbitmaps'],
      install_requires=['numpy','pandas'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)