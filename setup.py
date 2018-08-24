from setuptools import setup, find_packages

# import version
with open('genericFISTA/_version.py','r') as f:
    exec(f.read())
assert '__version__' in locals()

setup( name             = 'generic FISTA'
     , version          = __version__
     , description      = 'implementation of FISTA algorithms'
     , keywords         = [ 'accelerated gradient descent', 'fast gradient descent']
     , author           = 'Hsiou-Yuan Liu'
     , author_email     = 'hyliu@eecs.berkeley.edu'
     , license          = 'BSD'
     , packages         = find_packages()
     , py_modules       = ['genericFISTA']
     , install_requires = ['contexttimer']
     )
