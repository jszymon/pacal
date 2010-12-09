from distutils.core import setup

setup(
    name='PaCal',
    version='1.0',
    description='PaCal - ProbAbilistic CALculator',
    packages=['pacal','pacal.test'],
    author = 'Szymon Jaroszewicz, Marcin Korzen',
    author_email = 'mkorzen@wi.zut.edu.pl',
    license='GNU General Public License',
    url='http://pacal.sf.net',
    long_description=open('README.txt').read(),
)