mkdir docs
sphinx-quickstart docs
No separate source or build...
EBKAFramework
v 0.2




in conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('...'))
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

sphinx-apidoc -o docs/source/ .


ls -l docs/source | grep .rst | awk '{print $9}' | xargs -I % basename % ".rst" | awk '{print "source/"$1}'
copy to index.rst

cd docs
make builder
make html