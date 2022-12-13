# EdnaML

This directory contains a complete configuration YAML schema, configuration plus code samples for common experiments, we well as best practices and design patterns.

## Complete configuration YAML schema

The [config-full.yml](config-full.yml) file contains the complete configuration schema, with the default values EdnaML uses to populate empty configuration sections. See [config-full.md](config-full.md) for additional details on configuration sections.

## Common experiments

The `sample-configs` directory contains configurations and code for common ML experiments. For most experiments, we provide 2 ways to conduct them: 

1. The canonical method is to use configuration files to fully specify experiment details, such as classifier architecture, data loading, arguments, hyperparameters, etc and use the EdnaML class as a declarative API

2. The imperative method is to directly construct experiments using EdnaML basic building blocks. This is useful for research iteration and debugging when reproducibility and provenance are not high priority.

## Best practices and design

WIP








