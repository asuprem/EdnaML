docs:
	pdoc3 src/ednaml/ --html -o docs
	mv docs/ usage-docs/


.PHONY: docs