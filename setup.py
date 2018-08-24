from setuptools import setup, find_packages

# import version
with open('genericFISTA/_version.py','r') as f:
    exec(f.read())
assert '__version__' in locals()

setup( name             = 'Accelerated Proximal Gradient Descent'
     , version          = __version__
     , description      = 'implementation of APGC algorithm with optional restart feature'
     , keywords         = ['accelerated', 'proximal', 'gradient descent']
     , author           = 'Hsiou-Yuan Liu'
     , author_email     = 'hyhliu1989@gmail.com'
     , license          = 'BSD'
     , packages         = find_packages()
     , py_modules       = ['genericFISTA']
     , install_requires = ['contexttimer']
     )
