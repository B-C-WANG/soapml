from distutils.core import setup


package_dir = {"soapml":"soapml"}

packages = [
      "soapml",

]




setup(
      name="soapml",
      version="0.1",
      description="Machine learning with SOAP",
      author="B.C. WANG",
      url="https://github.com/B-C-WANG",
      license="LICENSE",
      package_dir=package_dir,
      packages=packages
      )