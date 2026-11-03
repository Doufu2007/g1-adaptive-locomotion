"""Create a portable source/evidence bundle without credentials or other runs."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=ROOT/'deliverables/g1_semantic_split_20261026.zip')
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.target.exists():
        raise FileExistsError('retain existing bundle; choose a new target')
    files = set()
    for pattern in ('src/**/*', 'scripts/*g1_semantic_split*.py', 'tests/test_g1_semantic_split*.py',
                    'docs/superpowers/*/*g1-semantic-split*.md', 'G1_SEMANTIC_SPLIT_README.md',
                    'requirements-semantic-split.txt', 'setup.py', 'LICENSE*',
                    'results/g1_semantic_split_20261026/**/*'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
    with zipfile.ZipFile(args.target, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
        for path in sorted(files):
            bundle.write(path, 'g1_semantic_split/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(args.target) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC verification failed: '+bad)
    receipt = {'file': args.target.name, 'bytes': args.target.stat().st_size, 'files': len(files),
               'sha256': hashlib.sha256(args.target.read_bytes()).hexdigest(), 'zip_crc_passed': True,
               'scope': 'source snapshot and this study evidence; no scientific training results claimed'}
    args.target.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
