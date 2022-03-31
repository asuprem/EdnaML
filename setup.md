sudo apt-get install python3.8-venv


create a dev virtualenv
pip3 install --upgrade build
pip3 install --upgrade twine

python3 -m build
python3 -m twine upload --repository testpypi dist/*

 pip3 install -e . <-- development install -->