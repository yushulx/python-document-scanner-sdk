from setuptools.command import build_ext
from setuptools import setup, Extension
import sys
import os
import io
from setuptools.command.install import install
import shutil
from pathlib import Path

ddn_lib_dir = ''
ddn_include = ''
ddn_lib_name = 'DynamsoftDocumentNormalizer'
core_lib_name = 'DynamsoftCore'

if sys.platform == "linux" or sys.platform == "linux2":
    # Linux
    ddn_lib_dir = 'lib/linux'
elif sys.platform == "win32":
    # Windows
    ddn_lib_name = 'DynamsoftDocumentNormalizerx64'
    core_lib_name = 'DynamsoftCorex64'
    ddn_lib_dir = 'lib/win'

if sys.platform == "linux" or sys.platform == "linux2":
    ext_args = dict(
        library_dirs=[ddn_lib_dir],
        extra_compile_args=['-std=c++11'],
        extra_link_args=["-Wl,-rpath=$ORIGIN"],
        libraries=[core_lib_name, ddn_lib_name, 'pthread'],
        include_dirs=['include']
    )


long_description = io.open("README.md", encoding="utf-8").read()

if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
    module_docscanner = Extension(
        'docscanner', ['src/docscanner.cpp'], **ext_args)
else:
    module_docscanner = Extension('docscanner',
                                  sources=['src/docscanner.cpp'],
                                  include_dirs=['include'], library_dirs=[ddn_lib_dir], libraries=[core_lib_name, ddn_lib_name])


def copyfiles(src, dst):
    if os.path.isdir(src):
        filelist = os.listdir(src)
        for file in filelist:
            libpath = os.path.join(src, file)
            shutil.copy2(libpath, dst)
    else:
        shutil.copy2(src, dst)

class CustomBuildExt(build_ext.build_ext):
    def run(self):
        build_ext.build_ext.run(self)
        dst = os.path.join(self.build_lib, "docscanner")
        copyfiles(ddn_lib_dir, dst)
        filelist = os.listdir(self.build_lib)
        for file in filelist:
            filePath = os.path.join(self.build_lib, file)
            if not os.path.isdir(file):
                copyfiles(filePath, dst)
                # delete file for wheel package
                os.remove(filePath)


class CustomBuildExtDev(build_ext.build_ext):
    def run(self):
        build_ext.build_ext.run(self)
        dev_folder = os.path.join(Path(__file__).parent, 'docscanner')
        copyfiles(ddn_lib_dir, dev_folder)
        filelist = os.listdir(self.build_lib)
        for file in filelist:
            filePath = os.path.join(self.build_lib, file)
            if not os.path.isdir(file):
                copyfiles(filePath, dev_folder)


class CustomInstall(install):
    def run(self):
        install.run(self)


setup(name='document-scanner-sdk',
      version='1.0.3',
      description='Document Scanner SDK for document edge detection, border cropping, perspective correction and brightness adjustment',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='yushulx',
      url='https://github.com/yushulx/python-document-scanner-sdk',
      license='MIT',
      packages=['docscanner'],
      ext_modules=[module_docscanner],
      classifiers=[
           "Development Status :: 5 - Production/Stable",
           "Environment :: Console",
           "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: MacOS",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: C++",
          "Programming Language :: Python :: Implementation :: CPython",
          "Topic :: Scientific/Engineering",
          "Topic :: Software Development",
      ],
      install_requires=['opencv-python'],
      entry_points={
          'console_scripts': ['scandocument=docscanner.scripts:scandocument']
      },
      cmdclass={
          'install': CustomInstall,
          'build_ext': CustomBuildExt,
          'develop': CustomBuildExtDev},
      )
