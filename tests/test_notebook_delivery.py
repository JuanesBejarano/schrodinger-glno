"""Check the delivered notebook without building it or training a model."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_syntax_and_embedded_imports(tmp_path):
    notebook = json.loads((ROOT / 'notebooks/GLNO_Shapes_Colab.ipynb').read_text())
    modules = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            header, _, body = source.partition('\n')
            relative = Path(header.removeprefix('%%writefile '))
            if relative.suffix == '.py':
                ast.parse(body)
                assert body == (ROOT / relative).read_text()
                modules.append('.'.join(relative.relative_to('src').with_suffix('').parts).removesuffix('.__init__'))
            else:
                assert json.loads(body) == json.loads((ROOT / relative).read_text())
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        else:
            ast.parse(source)
    script = f'import sys, importlib\nsys.path.insert(0, {str(tmp_path / "src")!r})\n'
    script += f'for name in {modules!r}:\n    assert importlib.import_module(name).__file__.startswith({str(tmp_path)!r})\n'
    script += 'assert not any(n.startswith("schrodinger_fno") for n in sys.modules)\n'
    subprocess.run([sys.executable, '-c', script], cwd=tmp_path, check=True)
