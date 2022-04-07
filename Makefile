docs:
	pdoc3 src/ednaml/  --force --html -o docs 

build:
	rm -f dist/*
	python3 -m build

upload:
	python3 -m twine upload --repository pypi dist/*

.PHONY: docs