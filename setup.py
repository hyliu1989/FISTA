#!/path/to/your/python
# The above line is usually  #!/usr/bin/python

from setuptools import setup

setup( name             = 'generic FISTA'
     , version          = '1.3.0'
     , description      = 'implementation of FISTA algorithms'
     , keywords         = [ 'accelerated gradient descent', 'fast gradient descent']
     , author           = 'Hsiou-Yuan Liu'
     , author_email     = 'hyliu@eecs.berkeley.edu'
     , license          = 'BSD'
     , packages         = ['genericFISTA']
     )