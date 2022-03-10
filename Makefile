docs:
	pdoc3 . --html -o docs
	mv docs/ usage-docs/


.PHONY: docs