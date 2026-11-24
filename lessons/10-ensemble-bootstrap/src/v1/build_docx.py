"""把提交目录里的 md 转成 docx：正文里的 Unicode 上下标换成 Word 真上下标，再调 pandoc。
用法: python -I build_docx.py <md 文件> [更多 md 文件]"""
import re, subprocess, sys, tempfile
from pathlib import Path
WORK = Path(__file__).resolve().parent
SUP = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁻':'−','⁺':'+','ᵀ':'T'}
SUBS = {'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9','₊':'+','₋':'−','ᵢ':'i'}
sup_re = re.compile('[' + ''.join(SUP) + ']+'); sub_re = re.compile('[' + ''.join(SUBS) + ']+')
def convert(text):
    out, in_code, in_yaml = [], False, False
    for n, line in enumerate(text.split('\n')):
        if n == 0 and line == '---': in_yaml = True; out.append(line); continue
        if in_yaml:
            if line == '---': in_yaml = False
            out.append(line); continue
        if line.startswith('```'): in_code = not in_code; out.append(line); continue
        if in_code or line.startswith('$$'): out.append(line); continue
        line = sup_re.sub(lambda m: '^' + ''.join(SUP[c] for c in m.group()) + '^', line)
        line = sub_re.sub(lambda m: '~' + ''.join(SUBS[c] for c in m.group()) + '~', line)
        out.append(line)
    return '\n'.join(out)
for arg in sys.argv[1:]:
    md = Path(arg); body = convert(md.read_text(encoding='utf8'))
    assert '~~' not in body, 'adjacent subscripts would become strikeout'
    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False, encoding='utf8', dir=WORK) as t: t.write(body); tmp = t.name
    out = md.with_suffix('.docx')
    subprocess.run(['pandoc', tmp, '-o', str(out), '--from', 'markdown', '--to', 'docx', f'--reference-doc={WORK/"reference-zh.docx"}', f'--resource-path={md.parent}'], check=True)
    Path(tmp).unlink(); print('built', out.name, out.stat().st_size)
