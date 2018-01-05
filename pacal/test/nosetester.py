"""
install nose package from http://pypi.python.org/pypi/nose/0.11.1
(see: http://somethingaboutorange.com/mrl/projects/nose/0.11.1/)
install coveraga package form http://pypi.python.org/pypi/coverage
"""
from __future__ import print_function

import nose
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #nose.run(argv=[__file__, '--verbosity=1', '--with-doctest', '--with-xunit', '--xunit-file=out/pacaltests.xml']) # it works ok. Following additional options: '--cover-html', '--cover-html-dir=out' cause errors
    #print nose.run(argv=[__file__, '-v', '-i=pacal/*', '--with-profile', '--profile-stats-file=out/pacal.proof']) # it works, but unknown output format (file abcd.txt)
    #nose.run(argv=[__file__, '-v'])
    #nose.run()
    print(nose.run(argv=[__file__, '--verbosity=2', '--with-id', '--with-coverage', '--cover-tests']))
    plt.show()
