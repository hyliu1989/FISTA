from setuptools import setup, find_packages

setup( name             = 'generic FISTA'
     , version          = '1.3.2'
     , description      = 'implementation of FISTA algorithms'
     , keywords         = [ 'accelerated gradient descent', 'fast gradient descent']
     , author           = 'Hsiou-Yuan Liu'
     , author_email     = 'hyliu@eecs.berkeley.edu'
     , license          = 'BSD'
     , packages         = find_packages()
     , py_modules       = ['genericFISTA']
     , install_requires = ['contexttimer']
     )
