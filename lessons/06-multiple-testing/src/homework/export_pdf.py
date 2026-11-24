"""把作业报告 Markdown 导出为带图的 PDF（不需要 pandoc）。

用法（需要 markdown-it-py，conda base 环境自带）：
    /opt/homebrew/Caskroom/miniconda/base/bin/python export_pdf.py
改完 .md（填姓名、勾选核对项）后重新运行即可。
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parents[1] / "report"  # lessons/06-multiple-testing/report
SRC = HERE / "第五课作业_多重检验校正.md"
HTML = HERE / "第五课作业_多重检验校正.html"
PDF = HERE / "第五课作业_多重检验校正.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "PingFang SC", "Hiragino Sans GB", sans-serif; font-size: 10.2pt;
       line-height: 1.6; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; border-bottom: 1.5px solid #2a78d6; padding-bottom: 2pt; margin-top: 16pt; }
h3 { font-size: 11pt; margin-top: 12pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.8pt; }
th, td { border: 1px solid #c3c2b7; padding: 2.5pt 4pt; vertical-align: top; }
th { background: #f1f0ea; }
tr { break-inside: avoid; }
img { display: block; max-width: 100%; margin: 6pt auto 2pt; break-inside: avoid; }
code { font-family: Menlo, monospace; font-size: 8.6pt; background: #f4f4f1; padding: 0 2pt; }
blockquote { color: #52514e; border-left: 3px solid #c3c2b7; margin: 6pt 0; padding: 0 8pt; }
hr { border: 0; border-top: 1px solid #e1e0d9; margin: 10pt 0; }
"""


def main():
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(SRC.read_text(encoding="utf-8"))
    body = body.replace("<li>[ ] ", "<li>☐ ").replace("<li>[x] ", "<li>☑ ")
    HTML.write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                    f"<title>{SRC.stem}</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={PDF}",
                        HTML.as_uri()], check=True, capture_output=True)
    print("已导出", PDF)


if __name__ == "__main__":
    sys.exit(main())
