"""
Configuration loader with support for YAML imports and Jinja2 templating.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, BaseLoader


def load_yaml_with_imports(file_path: Path, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load a YAML file with support for !import directives and Jinja2 templating.
    
    Args:
        file_path: Path to the YAML file
        context: Optional context dictionary for Jinja2 templating
        
    Returns:
        Dictionary with the loaded and processed configuration
    """
    # Set up Jinja2 environment
    env = Environment(loader=BaseLoader())
    
    # Add custom filters
    env.filters['to_yaml'] = lambda x: yaml.dump(x, default_flow_style=False)
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Render the template if context is provided
    if context is not None:
        template = env.from_string(content)
        content = template.render(**context)
    
    # Parse the YAML content
    config = yaml.safe_load(content) or {}
    
    # Handle imports if present
    if 'imports' in config:
        base_dir = file_path.parent
        imported_data = {}
        
        for import_path in config['imports']:
            # Handle relative paths
            if not os.path.isabs(import_path):
                import_path = base_dir / import_path
            else:
                import_path = Path(import_path)
            
            # Recursively load the imported file
            imported = load_yaml_with_imports(import_path, context)
            imported_data.update(imported)
        
        # Merge the imported data with the current config
        # Local config overrides imported values
        config = {**imported_data, **config}
        
        # Remove the imports key as it's no longer needed
        config.pop('imports', None)
    
    return config


def resolve_metrics(metrics_config: Dict[str, Any], metric_groups: List[str]) -> List[str]:
    """
    Resolve metric groups to individual metric names.
    
    Args:
        metrics_config: Dictionary containing metric groups
        metric_groups: List of metric names or group references
        
    Returns:
        List of resolved metric names
    """
    resolved_metrics = []
    for item in metric_groups:
        # If the item is a reference to a group (starts with @)
        if isinstance(item, str) and item.startswith('@'):
            group_name = item[1:]
            entries = metrics_config.get(group_name, [])
            if isinstance(entries, list):
                resolved_metrics.extend(resolve_metrics(metrics_config, entries))
        else:
            # Direct metric reference or dict entry
            if isinstance(item, dict) and 'name' in item:
                resolved_metrics.append(item['name'])
            elif isinstance(item, str):
                resolved_metrics.append(item)
            else:
                # Skip unsupported item types
                continue
    # Remove duplicates while preserving order
    seen = set()
    return [m for m in resolved_metrics if not (m in seen or seen.add(m))]
