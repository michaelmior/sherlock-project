from collections import defaultdict
import os
import sys

from pydeps import py2depgraph, cli
from pydeps.target import Target
import ruamel.yaml


def collect_deps(path, dep_graph, deps=None):
    if deps is None:
        deps = set()

    for dep in dep_graph[path].imports:
        if dep not in deps:
            deps.add(dep)
            collect_deps(dep, dep_graph, deps)

    return deps

if __name__ == '__main__':
    # Generate the dependency graph
    options = cli.parse_args(["sherlock", "--noshow", "--only", "sherlock"])
    target = Target(options["fname"])
    with target.chdir_work():
        dep_graph = py2depgraph.py2dep(target, **options)

    script_deps = {}
    for script in os.listdir(os.path.join(os.path.dirname(__file__), 'sherlock', 'scripts')):
        # Calculate both file path and import path
        script_path = os.path.join('sherlock', 'scripts', script)
        script_name = script_path.replace(os.sep, '.')[:-3]

        # Skip over __init__ in scripts directory
        if script_name == 'sherlock.scripts.__init__':
            continue

        # Recursively gather internal dependencies for the script
        script_deps[script_name] = [os.path.relpath(dep_graph[d].path) for d in collect_deps(script_name, dep_graph)]
        script_deps[script_name].append(script_path)

    # Load the existing dvc.yaml
    yaml = ruamel.yaml.YAML() 
    dvc_yaml_path = os.path.join(os.path.dirname(__file__), 'dvc.yaml')
    dvc_yaml = yaml.load(open(dvc_yaml_path).read())

    # Update the internal Python dependencies
    for (script_name, deps) in script_deps.items():
        stage_yaml = dvc_yaml['stages'][script_name.split('.')[-1]]
        print(stage_yaml)
        if 'deps' in stage_yaml:
            stage_yaml['deps'] = [d for d in stage_yaml['deps'] if not d.startswith('sherlock')]
        if deps and 'deps' not in stage_yaml:
            stage_yaml['deps'] = []
        stage_yaml['deps'] += deps

    yaml.dump(dvc_yaml, sys.stdout)
