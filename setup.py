from setuptools import setup

setup(
        name='skensemble',
        version='0.0.1',
        author='iwatobipen',
        author_email='seritaka@gmail.com',
        packages=['skensemble'],
        url='https://github.com/iwatobipen/skensemble',
        description='blending framework for python',
        long_description=read_md('README.md'),
        install_requires=['numpy',
                          'scikit-learn'],
        classifiers=[
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.2',
                    'Programming Language :: Python :: 3.3',
                    'Programming Language :: Python :: 3.4',
                    'Programming Language :: Python :: 3.5',  
                    'Programming Language :: Python :: 3.6'
            ],
        )
