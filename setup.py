from setuptools import setup

setup(
    name='CTAB-GAN',
    version='1.0',
    description='CTAB-GAN',
    url='https://github.com/XinGuu/CTAB-GAN',
    author='Anonymous',
    license='MIT',
    packages=['model'],
    zip_safe=False,
    install_requires=['numpy==1.21.0', 'torch==1.9.1', 'pandas==1.2.4', 'scikit-learn==0.24.1',
                      'dython==0.6.4.post1', 'scipy==1.4.1'],
)