import subprocess

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


def custom_command():
    subprocess.call(["pip", "install", "-r", "requirements.txt", "--user"])


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


setup(
    name="image-quality-assessment",
    version='0.0.0.0',
    description="",
    zip_safe=True,
    url="https://github.com/rafaelpadilla/TCF-LMO",
    packages=["src"],
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)
