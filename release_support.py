"""Portable configuration shared by the released research entry points."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import re
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs' / 'release.yaml'

def resolve_path(value, base):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()

def load_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in config.get('paths', {}).items():
        if isinstance(value, str) and value:
            config['paths'][key] = str(resolve_path(value, path.parent))
    return config

def config_path_from_argv():
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--config' and i + 1 < len(sys.argv):
            return resolve_path(sys.argv[i + 1], Path.cwd())
        if arg.startswith('--config='):
            return resolve_path(arg.split('=', 1)[1], Path.cwd())
    return DEFAULT_CONFIG

def release_paths(section):
    path = config_path_from_argv()
    config = load_yaml(path)
    settings = config.get('pipelines', {}).get(section, {})
    paths = dict(config.get('paths', {}))
    for key, value in settings.items():
        if key.endswith('_dir') or key.endswith('_root'):
            paths[key] = str(resolve_path(value, path.parent))
        else:
            paths[key] = value
    return paths

def apply_paths(namespace, section):
    paths = release_paths(section)
    namespace['DATA_ROOT'] = Path(paths.get('original_root', ROOT))
    namespace['EMBEDDING_ROOT'] = Path(paths.get('embedding_root', ROOT / 'baselines/text/modern_embedding_run'))
    shared = sys.modules.get('run_dynamic_drift_generation')
    if shared is not None:
        shared.DATA_ROOT = namespace['DATA_ROOT']
        shared.EMBEDDING_ROOT = namespace['EMBEDDING_ROOT']
    if 'STATIC_RUN' in namespace:
        namespace['STATIC_RUN'] = Path(paths['static_run'])
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == '--input-dir' and i + 1 < len(sys.argv):
                namespace['STATIC_RUN'] = resolve_path(sys.argv[i + 1], Path.cwd())
            elif arg.startswith('--input-dir='):
                namespace['STATIC_RUN'] = resolve_path(arg.split('=', 1)[1], Path.cwd())
    if 'output_dir' in paths:
        out = Path(paths['output_dir'])
        for key in ['RESULTS_DIR', 'RESULT_DIR', 'RESULTS']:
            if key in namespace: namespace[key] = out
        for key in ['FIGURES_DIR', 'FIG_DIR', 'FIGURES']:
            if key in namespace: namespace[key] = out / 'figures'
    return paths

def parse_release_args(parser, namespace, section):
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Release YAML; paths are relative to this file.')
    args = parser.parse_args()
    paths = apply_paths(namespace, section)
    args.config = str(Path(args.config).resolve())
    args.runtime_config = paths.get('runtime_config', str(ROOT / 'RedditBotSim/config.yml'))
    for key in ['input_dir', 'output_dir', 'openai_model', 'gemini_model']:
        flag = '--' + key.replace('_', '-')
        explicit = any(a == flag or a.startswith(flag + '=') for a in sys.argv[1:])
        if hasattr(args, key) and key in paths and not explicit:
            setattr(args, key, paths[key])
        if key.endswith('_model') and hasattr(args, key) and not explicit:
            setattr(args, key, os.environ.get(key.upper(), getattr(args, key)))
    for key in ['input_dir', 'output_dir']:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, str(resolve_path(getattr(args, key), Path.cwd())))
    return args

def research_cli(namespace, section):
    parser = argparse.ArgumentParser(description=namespace.get('__doc__'))
    parser.add_argument('--input-dir', type=Path, help='Override the static family dataset directory.')
    parser.add_argument('--output-dir', type=Path, help='Override this evaluation output directory.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Release YAML; paths are file-relative.')
    args = parser.parse_args()
    if args.input_dir is not None: namespace['STATIC_RUN'] = Path(args.input_dir).resolve()
    if args.output_dir is not None:
        old_root = Path(release_paths(section)['output_dir'])
        new_root = Path(args.output_dir).resolve()
        def rebase(value):
            if isinstance(value, Path):
                try: return new_root / value.relative_to(old_root)
                except ValueError: return value
            if isinstance(value,dict): return {k:rebase(v) for k,v in value.items()}
            if isinstance(value,list): return [rebase(v) for v in value]
            if isinstance(value,tuple): return tuple(rebase(v) for v in value)
            return value
        for key in list(namespace):
            if key.isupper(): namespace[key] = rebase(namespace[key])
    return args

def redact_error(error):
    text = str(error)
    for key, value in os.environ.items():
        if value and len(value) >= 8 and any(t in key.upper() for t in ['API_KEY','TOKEN','SECRET']):
            text = text.replace(value, '[REDACTED]')
    text = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+', r'\1[REDACTED]', text)
    return text
