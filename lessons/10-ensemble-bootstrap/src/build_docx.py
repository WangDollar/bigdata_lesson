"""把作业报告（Markdown）生成 Word 版。

做五件事：
1. 把报告里三张字符画（分析主线、集成学习框架、深度 2 的树）重画成图片：Word 里中英文字符宽度不同，字符画会错位。
2. 生成带中文字体的 Word 模板：正文宋体 + Times New Roman，标题黑体，A4，三线表，页眉页码，目录样式。
3. 调用 pandoc：H1 作为文档标题、图片按页面宽度缩放；Lua 过滤器把引用块（说明、我的追问）变成带底色的提示框。Markdown 原文不改。
4. 排版后处理：封面 → 目录 → 正文，每章另起一页；表格按内容分配列宽、数字列居中、
   小表不跨页、长表跨页时重复表头；图片居中并编号。
5. 本机装了 Microsoft Word 时，用它把目录和页码真正生成出来（WPS 不会自动刷新目录）；
   否则退回为“打开时提示更新域”。

运行：python src/build_docx.py
"""
import io
import math
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parents[1]
MD = BASE / "report" / "作业报告_集成学习与Bootstrapping_CTG.md"
DOCX = MD.with_suffix(".docx")
FIG = BASE / "outputs" / "report"
INK, INK2 = "#0b0b0b", "#52514e"
KIND = {"blue": ("#e8f1fc", "#2a78d6"), "orange": ("#fdeee7", "#eb6834"), "gray": ("#f3f2ef", "#a8a69f"), "white": ("#ffffff", "#898781")}
plt.rcParams.update({"font.family": ["PingFang SC"], "axes.unicode_minus": False})


# ---------- 1. 三张示意图 ----------
def box(ax, x, y, w, h, title, body="", kind="white", size=10):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=1.4))
    if body:
        ax.text(x, y + h / 2 - 1.6, title, ha="center", va="top", fontsize=size + 0.5, fontweight=600, color=INK)
        ax.text(x, y + h / 2 - 4.3, body, ha="center", va="top", fontsize=size - 0.5, color=INK2, linespacing=1.45)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=size + 0.5, fontweight=600, color=INK)


def elbow(ax, x0, y0, x1, y1):
    ym = (y0 + y1) / 2
    ax.plot([x0, x0, x1, x1], [y0, ym, ym, y1], color="#898781", lw=1.2, zorder=0)


def draw_framework(path):
    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    box(ax, 50, 59, 46, 6, "集成学习：把多个模型的结果组合起来", kind="white", size=12)
    l2 = [(16, 29, "Bagging 路线", "并行 · 平均 · 主要降方差", "blue"),
          (47, 29, "Boosting 路线", "顺序 · 纠错 · 主要降偏差", "orange"),
          (73, 18, "Stacking（堆叠）", "元模型学“该听谁的”\n本课未展开", "gray"),
          (91.5, 15, "级联 Cascade", "逐级筛选、提前判定\n本课未展开", "gray")]
    for x, w, t, b, k in l2:
        elbow(ax, 50, 56, x, 51)
        box(ax, x, 46, w, 10, t, b, k)
    l3 = [(7.5, 14, "Bagging", "bootstrap 抽样本\n各训练一棵深树\n再取平均", "blue", 16),
          (23, 14, "随机森林", "bagging +\n每个节点随机\n挑部分特征", "blue", 16),
          (39, 14, "AdaBoost", "调高分错样本\n的权重\n最终加权投票", "orange", 47),
          (55, 14, "梯度提升", "新树拟合\n负梯度 y − p\n逐轮累加", "orange", 47)]
    for x, w, t, b, k, px in l3:
        elbow(ax, px, 41, x, 37)
        box(ax, x, 30.5, w, 13, t, b, k)
    elbow(ax, 55, 24, 55, 21)
    box(ax, 55, 16.5, 19, 9, "XGBoost / LightGBM", "正则化 + 二阶导 + 工程加速", "orange")
    box(ax, 23, 4.5, 44, 8, "基学习器：决策树", "Bagging 用深树（偏差低、方差高）；Boosting 用浅树或树桩", "white")
    box(ax, 76, 4.5, 46, 8, "Bootstrap：一种独立的重抽样方法，不是模型", "① 给 Bagging 制造多份训练集　② 衡量稳定性与不确定性", "white")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_mainline(path):
    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 23)
    ax.axis("off")
    parts = [("第一篇　问题：单棵树", "浅树偏差大，深树方差大\n不稳定集中在病理片段", "偏差² 0.056 → 0.015\n方差占比 1/10 → 2/5", "gray"),
             ("第二篇　Bagging 路线", "并行平均去掉方差\n随机森林再压低树间相关 ρ", "深树方差去掉约 3/4\nρ 0.23 → 0.04", "blue"),
             ("第三篇　Boosting 路线", "顺序叠加去掉偏差\n轮数过多会过拟合", "同样 100 个树桩\n偏差²降约 70%", "orange"),
             ("第四篇　实战与边界", "公平比较 · 评价协议\n阈值取舍 · 特征重要性", "GB ≈ RF > LR，都远好于单树\n按行评价会选错冠军", "gray")]
    xs = [12, 37.3, 62.6, 88]
    for x, (t, b, note, k) in zip(xs, parts):
        box(ax, x, 16, 22, 11, t, b, k, size=10.5)
        ax.text(x, 8.6, note, ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.45)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - 11.3, 16), xytext=(x0 + 11.3, 16),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=0, shrinkB=0))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_tree(path):
    """4.1 深度 2 的树，自上而下画。数字与报告原文的字符画一致。"""
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")

    def link(x0, y0, x1, y1, label):
        elbow(ax, x0, y0, x1, y1)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.6, label, ha="center", va="bottom", fontsize=9.5, color=INK2)

    box(ax, 50, 40, 20, 6, "Mode ≤ 106？")
    for x, cond in ((25, "Max ≤ 234？"), (75, "ALTV ≤ 68.5？")):
        link(50, 37, x, 30, "是" if x < 50 else "否")
        box(ax, x, 27, 21, 6, cond)
    leaves = [(12.5, 25, "判病理", "病理 54 / 其余 3", ""),
              (37.5, 25, "判其余", "病理 0 / 其余 2", "↑ 这次切分只用来\n分开 2 个样本"),
              (62.5, 75, "判其余", "病理 29 / 其余 1164", "↑ 29 个病理片段\n被漏掉"),
              (87.5, 75, "判病理", "病理 25 / 其余 2", "")]
    for x, parent, verdict, counts, note in leaves:
        link(parent, 24, x, 17, "是" if x < parent else "否")
        box(ax, x, 12.5, 22, 9, "", kind="orange" if verdict == "判病理" else "gray")
        ax.text(x, 14.3, verdict, ha="center", va="center", fontsize=10.5, fontweight=600, color=INK)
        ax.text(x, 10.6, counts, ha="center", va="center", fontsize=9.5, color=INK2)
        if note:
            ax.text(x, 4.2, note, ha="center", va="center", fontsize=9.5, color=KIND["orange"][1], linespacing=1.4)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- 2. Word 模板 ----------
BODY_FONT, BODY_EA, HEAD_FONT, HEAD_EA = "Times New Roman", "SimSun", "Arial", "SimHei"
PAGE_W, MARGIN_LR = 11906, 1247
TEXT_W = PAGE_W - 2 * MARGIN_LR                     # 版心宽度（twips）
ACCENT = "2A78D6"                                   # 与示意图里的蓝色一致


def rfonts(a, ea):
    return f'<w:rFonts w:ascii="{a}" w:hAnsi="{a}" w:eastAsia="{ea}" w:cs="{a}" />'


HEAD_RFONTS = rfonts(HEAD_FONT, HEAD_EA)


def edit_style(xml, sid, fn):
    pat = re.compile(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, re.S)
    assert pat.search(xml), sid
    return pat.sub(lambda m: fn(m.group(0)), xml, count=1)


def upsert_style(xml, sid, block):
    """有这个样式就整段替换，没有就追加。"""
    pat = re.compile(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, re.S)
    if pat.search(xml):
        return pat.sub(lambda m: block, xml, count=1)
    return xml.replace("</w:styles>", block + "\n</w:styles>")


def heading(sz):
    def fn(block):
        block = re.sub(r"<w:rFonts[^>]*/>", HEAD_RFONTS, block)
        block = re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000" />', block)
        if sz:
            block = re.sub(r'<w:sz w:val="\d+" ?/>', f'<w:sz w:val="{sz}" />', block)
            block = re.sub(r'<w:szCs w:val="\d+" ?/>', f'<w:szCs w:val="{sz}" />', block)
        return block
    return fn


def pstyle(sid, name, ppr, rpr="", based="Normal", extra=""):
    return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}" /><w:basedOn w:val="{based}" />{extra}'
            f"<w:qFormat /><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>")


def sz(half_points):
    return f'<w:sz w:val="{half_points}" /><w:szCs w:val="{half_points}" />'


STYLES = {
    # 正文两端对齐（中文排版习惯）
    "BodyText": pstyle("BodyText", "Body Text", '<w:spacing w:before="60" w:after="60" /><w:jc w:val="both" />', extra='<w:link w:val="BodyTextChar" />'),
    # 封面：标题往下放，副标题灰色
    "Title": pstyle("Title", "Title", '<w:spacing w:before="2400" w:after="240" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
                    HEAD_RFONTS + "<w:b /><w:bCs />" + sz(44)),
    "Subtitle": pstyle("Subtitle", "Subtitle", '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
                       HEAD_RFONTS + '<w:color w:val="404040" />' + sz(30)),
    # 一级标题：每章另起一页，下加细线
    "Heading1": pstyle("Heading1", "heading 1",
                       '<w:keepNext /><w:keepLines /><w:pageBreakBefore />'
                       '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="808080" /></w:pBdr>'
                       '<w:spacing w:before="0" w:after="240" w:line="240" w:lineRule="auto" /><w:outlineLvl w:val="0" />',
                       HEAD_RFONTS + '<w:b /><w:bCs /><w:color w:val="000000" />' + sz(32), extra='<w:next w:val="BodyText" />'),
    "Heading2": pstyle("Heading2", "heading 2",
                       '<w:keepNext /><w:keepLines /><w:spacing w:before="320" w:after="120" w:line="240" w:lineRule="auto" /><w:outlineLvl w:val="1" />',
                       HEAD_RFONTS + '<w:b /><w:bCs /><w:color w:val="000000" />' + sz(26), extra='<w:next w:val="BodyText" />'),
    # 目录：单独一页，标题居中，条目带点线和页码
    "TOCHeading": pstyle("TOCHeading", "TOC Heading",
                         '<w:keepNext /><w:pageBreakBefore /><w:spacing w:before="0" w:after="240" /><w:jc w:val="center" /><w:outlineLvl w:val="9" />',
                         HEAD_RFONTS + "<w:b /><w:bCs />" + sz(32)),
    "TOC1": pstyle("TOC1", "toc 1", f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_W - 10}" /></w:tabs><w:spacing w:before="90" w:after="0" w:line="264" w:lineRule="auto" />',
                   HEAD_RFONTS + "<w:b /><w:bCs />", extra='<w:next w:val="Normal" /><w:uiPriority w:val="39" /><w:unhideWhenUsed />'),
    "TOC2": pstyle("TOC2", "toc 2", f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_W - 10}" /></w:tabs><w:spacing w:before="0" w:after="0" w:line="264" w:lineRule="auto" /><w:ind w:left="420" />',
                   "", extra='<w:next w:val="Normal" /><w:uiPriority w:val="39" /><w:unhideWhenUsed />'),
    # 图片和图注居中；图片段落用单倍行距，否则多倍行距会在图上方多留空
    "Caption": pstyle("Caption", "caption", '<w:keepLines /><w:spacing w:before="60" w:after="240" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
                      '<w:color w:val="404040" />' + sz(19)),
    "CaptionedFigure": pstyle("CaptionedFigure", "Captioned Figure", '<w:keepNext /><w:spacing w:before="160" w:after="0" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
                              based="Figure"),
    # 页眉页脚
    "Header": pstyle("Header", "header",
                     f'<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="4" w:color="A6A6A6" /></w:pBdr><w:tabs><w:tab w:val="right" w:pos="{TEXT_W}" /></w:tabs>'
                     '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto" />',
                     '<w:color w:val="7F7F7F" />' + sz(17), extra='<w:uiPriority w:val="99" /><w:unhideWhenUsed />'),
    "Footer": pstyle("Footer", "footer", '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
                     '<w:color w:val="595959" />' + sz(18), extra='<w:uiPriority w:val="99" /><w:unhideWhenUsed />'),
    # 代码块：8 磅、浅灰底加细框，保证附录里最长的一行不折行
    "SourceCode": f"""<w:style w:type="paragraph" w:customStyle="1" w:styleId="SourceCode">
    <w:name w:val="Source Code" />
    <w:basedOn w:val="Normal" />
    <w:link w:val="VerbatimChar" />
    <w:pPr>
      <w:pBdr>
        <w:top w:val="single" w:sz="4" w:space="4" w:color="D9D9D9" /><w:left w:val="single" w:sz="4" w:space="4" w:color="D9D9D9" />
        <w:bottom w:val="single" w:sz="4" w:space="4" w:color="D9D9D9" /><w:right w:val="single" w:sz="4" w:space="4" w:color="D9D9D9" />
      </w:pBdr>
      <w:shd w:val="clear" w:color="auto" w:fill="F7F7F5" />
      <w:autoSpaceDE w:val="0" /><w:autoSpaceDN w:val="0" />
      <w:spacing w:before="120" w:after="160" w:line="240" w:lineRule="auto" />
      <w:ind w:left="100" w:right="100" />
      <w:jc w:val="left" />
    </w:pPr>
    <w:rPr>{rfonts("Consolas", BODY_EA)}{sz(16)}</w:rPr>
  </w:style>""",
    # 提示框（引用块）：左侧蓝色竖条 + 浅底色，字号略小于正文
    "CalloutTable": f"""<w:style w:type="table" w:customStyle="1" w:styleId="CalloutTable">
    <w:name w:val="Callout Table" />
    <w:basedOn w:val="TableNormal" />
    <w:rPr>{sz(21)}</w:rPr>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="nil" /><w:left w:val="single" w:sz="24" w:space="0" w:color="{ACCENT}" />
        <w:bottom w:val="nil" /><w:right w:val="nil" /><w:insideH w:val="nil" /><w:insideV w:val="nil" />
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="110" w:type="dxa" /><w:left w:w="220" w:type="dxa" /><w:bottom w:w="110" w:type="dxa" /><w:right w:w="220" w:type="dxa" />
      </w:tblCellMar>
    </w:tblPr>
    <w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="F2F6FB" /></w:tcPr>
  </w:style>""",
}

TABLE_STYLE = """<w:style w:type="table" w:default="1" w:styleId="Table">
    <w:name w:val="Table" />
    <w:basedOn w:val="TableNormal" />
    <w:qFormat />
    <w:rPr><w:sz w:val="19" /><w:szCs w:val="19" /></w:rPr>
    <w:tblPr>
      <w:tblInd w:w="0" w:type="dxa" />
      <w:tblBorders>
        <w:top w:val="single" w:sz="12" w:space="0" w:color="262626" />
        <w:left w:val="nil" />
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="262626" />
        <w:right w:val="nil" />
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9D9D9" />
        <w:insideV w:val="nil" />
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="30" w:type="dxa" /><w:left w:w="90" w:type="dxa" />
        <w:bottom w:w="30" w:type="dxa" /><w:right w:w="90" w:type="dxa" />
      </w:tblCellMar>
    </w:tblPr>
    <w:tblStylePr w:type="firstRow">
      <w:rPr><w:b /><w:bCs /></w:rPr>
      <w:tcPr>
        <w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="262626" /></w:tcBorders>
        <w:shd w:val="clear" w:color="auto" w:fill="F2F2F2" />
        <w:vAlign w:val="bottom" />
      </w:tcPr>
    </w:tblStylePr>
  </w:style>"""


def build_reference(path):
    src = Path(tempfile.mkdtemp()) / "ref.docx"
    subprocess.run(["pandoc", "-o", str(src), "--print-default-data-file", "reference.docx"], check=True)
    with zipfile.ZipFile(src) as zin:
        files = {n: zin.read(n) for n in zin.namelist()}
    s = files["word/styles.xml"].decode("utf-8")
    s = re.sub(r"(<w:rPrDefault>\s*<w:rPr>\s*)<w:rFonts[^>]*/>", r"\1" + rfonts(BODY_FONT, BODY_EA), s)
    s = re.sub(r'(<w:rPrDefault>.*?)<w:sz w:val="24" />\s*<w:szCs w:val="24" />',
               r'\1<w:sz w:val="22" /><w:szCs w:val="22" />', s, flags=re.S)
    s = re.sub(r'<w:pPrDefault>\s*<w:pPr>\s*<w:spacing w:after="200" />',
               '<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="312" w:lineRule="auto" />', s)
    for sid, size in (("Heading3", 23), ("Heading4", 22)):
        s = edit_style(s, sid, heading(size))
    for sid, block in STYLES.items():
        s = upsert_style(s, sid, block)
    s = edit_style(s, "Table", lambda b: TABLE_STYLE)
    s = edit_style(s, "VerbatimChar", lambda b: re.sub(r'<w:sz(?:Cs)? w:val="\d+" ?/>', "", re.sub(r"<w:rFonts[^>]*/>", rfonts("Consolas", BODY_EA), b)))
    files["word/styles.xml"] = s.encode("utf-8")
    d = files["word/document.xml"].decode("utf-8")
    d = re.sub(r"<w:sectPr>.*?</w:sectPr>",
               '<w:sectPr><w:footnotePr><w:numRestart w:val="eachSect" /></w:footnotePr>'
               f'<w:pgSz w:w="{PAGE_W}" w:h="16838" />'
               f'<w:pgMar w:top="1418" w:right="{MARGIN_LR}" w:bottom="1418" w:left="{MARGIN_LR}" w:header="851" w:footer="851" w:gutter="0" />'
               "</w:sectPr>", d, flags=re.S)
    files["word/document.xml"] = d.encode("utf-8")
    write_zip(path, files)


def write_zip(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, b in files.items():
            zout.writestr(n, b)


# 引用块 → 单格表格（Word 里能把整段内容连同列表一起框进同一个底色框）
CALLOUT_LUA = """
function BlockQuote(el)
  local body = {attr = pandoc.Attr(), body = {pandoc.Row({pandoc.Cell(el.content)})}, head = {}, row_head_columns = 0}
  return pandoc.Table({long = {}, short = {}}, {{pandoc.AlignDefault, 1.0}}, pandoc.TableHead(), {body}, pandoc.TableFoot(),
                      pandoc.Attr("", {"callout"}))
end
"""


# ---------- 4. 排版后处理（document.xml） ----------
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# OOXML 对子元素顺序有要求，顺序错了 Word 会报文件损坏
PPR_ORDER = ("pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs "
             "suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd "
             "snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment "
             "textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange").split()
TBLPR_ORDER = ("tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd "
               "tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription").split()
TCPR_ORDER = "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark".split()
RPR_ORDER = ("rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish "
             "webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang "
             "eastAsianLayout specVanish oMath").split()
SECT_ORDER = ("headerReference footerReference footnotePr endnotePr type pgSz pgMar paperSrc pgBorders lnNumType pgNumType "
              "cols formProt vAlign noEndnote titlePg textDirection bidi rtlGutter docGrid printerSettings").split()


def w(tag):
    return f"{{{W_NS}}}{tag}"


def local(el):
    return el.tag.rsplit("}", 1)[-1]


def put(parent, tag, order, attrs=None):
    """按规定顺序在 parent 下放一个子元素（同名旧元素先删掉）。"""
    for old in parent.findall(w(tag)):
        parent.remove(old)
    el = ET.Element(w(tag), {w(k): str(v) for k, v in (attrs or {}).items()})
    rank = order.index(tag)
    pos = next((i for i, c in enumerate(parent) if local(c) in order and order.index(local(c)) > rank), len(parent))
    parent.insert(pos, el)
    return el


def first_child(parent, tag):
    el = parent.find(w(tag))
    if el is None:
        el = ET.Element(w(tag))
        parent.insert(0, el)
    return el


def ppr(p):
    return first_child(p, "pPr")


def style_of(p):
    s = p.find(f"{w('pPr')}/{w('pStyle')}")
    return s.get(w("val")) if s is not None else ""


def set_spacing(p, **kw):
    sp = ppr(p).find(w("spacing"))
    if sp is None:
        sp = put(ppr(p), "spacing", PPR_ORDER)
    for k, v in kw.items():
        sp.set(w(k), str(v))
    if "line" in kw:
        sp.set(w("lineRule"), "auto")


def text_of(el):
    return "".join(t.text or "" for t in el.iter(w("t")))


def em_width(s):
    """按 Times New Roman / 宋体估算显示宽度（单位：字号 em）。"""
    total = 0.0
    for ch in s:
        o = ord(ch)
        if ch == " ":
            total += 0.25
        elif "A" <= ch <= "Z":
            total += 0.66
        elif o < 128:
            total += 0.5
        elif 0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
            total += 1.0
        else:
            total += 0.6
    return total


def longest_token(s):
    return max((em_width(t) for t in re.findall(r"[\x21-\x7e]+", s)), default=0.0)


NUMERIC = re.compile(r"^\s*(?:[+\-−–±]?\s?\d|—|–|-?$)")
TABLE_EM, CELL_PAD = 190, 220                      # 表格字号 9.5 磅 = 190 twips；左右单元格边距 + 余量


def allocate(need, minw, weight, total):
    """短列给足不换行的宽度，剩下的按平均内容长度分给长列，再保证每列不低于最长英文词宽。"""
    n = len(need)
    if sum(need) <= total:
        return [x + (total - sum(need)) * x / sum(need) for x in need]
    fixed = {}
    while True:
        free = [c for c in range(n) if c not in fixed]
        share = (total - sum(fixed.values())) / len(free)
        newly = [c for c in free if need[c] <= share]
        if not newly:
            break
        fixed.update({c: need[c] for c in newly})
    free = [c for c in range(n) if c not in fixed]
    rem = total - sum(fixed.values())
    wsum = sum(weight[c] for c in free) or 1
    width = dict(fixed)
    width.update({c: rem * weight[c] / wsum for c in free})
    short = [c for c in free if width[c] < minw[c]]
    if short:
        give = sum(minw[c] - width[c] for c in short)
        donors = [c for c in free if c not in short]
        slack = sum(width[c] - minw[c] for c in donors)
        for c in short:
            width[c] = minw[c]
        for c in donors:
            width[c] -= give * (width[c] - minw[c]) / slack if slack > 0 else 0
    return [width[c] for c in range(n)]


def format_callout(tbl):
    tblPr = tbl.find(w("tblPr"))
    for c in list(tblPr):
        tblPr.remove(c)
    put(tblPr, "tblStyle", TBLPR_ORDER, {"val": "CalloutTable"})
    put(tblPr, "tblW", TBLPR_ORDER, {"w": TEXT_W, "type": "dxa"})
    put(tblPr, "tblInd", TBLPR_ORDER, {"w": 0, "type": "dxa"})
    borders = put(tblPr, "tblBorders", TBLPR_ORDER)
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        attrs = {"val": "single", "sz": 24, "space": 0, "color": ACCENT} if side == "left" else {"val": "nil"}
        ET.SubElement(borders, w(side), {w(k): str(v) for k, v in attrs.items()})
    put(tblPr, "tblLayout", TBLPR_ORDER, {"type": "fixed"})
    mar = put(tblPr, "tblCellMar", TBLPR_ORDER)
    for side, v in (("top", 110), ("left", 220), ("bottom", 110), ("right", 220)):
        ET.SubElement(mar, w(side), {w("w"): str(v), w("type"): "dxa"})
    put(tblPr, "tblLook", TBLPR_ORDER, {"val": "0000", "firstRow": 0, "lastRow": 0, "firstColumn": 0, "lastColumn": 0, "noHBand": 1, "noVBand": 1})
    for g in tbl.find(w("tblGrid")):
        g.set(w("w"), str(TEXT_W))
    for tr in tbl.findall(w("tr")):
        put(first_child(tr, "trPr"), "cantSplit", ["cantSplit"])
        for tc in tr.findall(w("tc")):
            tcPr = first_child(tc, "tcPr")
            put(tcPr, "tcW", TCPR_ORDER, {"w": TEXT_W, "type": "dxa"})
            put(tcPr, "shd", TCPR_ORDER, {"val": "clear", "color": "auto", "fill": "F2F6FB"})
            ps = tc.findall(w("p"))
            set_spacing(ps[0], before=0)
            set_spacing(ps[-1], after=0)
            if not text_of(tbl).startswith("关于本报告的说明"):     # 追问、公式提示框与下文（回答）放在同一页
                for p in tc.iter(w("p")):
                    put(ppr(p), "keepNext", PPR_ORDER)


def format_table(tbl):
    rows = tbl.findall(w("tr"))
    grid = tbl.find(w("tblGrid")).findall(w("gridCol"))
    ncols = len(grid)
    cells = [tr.findall(w("tc")) for tr in rows]
    assert all(len(r) == ncols for r in cells), "表格里有合并单元格，列宽计算需要改"
    paras = [[[text_of(p) for p in tc.iter(w("p"))] for tc in r] for r in cells]
    body = paras[1:]

    # 列宽
    need = [max(em_width(t) for r in paras for t in r[c]) * TABLE_EM + CELL_PAD for c in range(ncols)]
    minw = [max(longest_token(t) for r in paras for t in r[c]) * TABLE_EM + CELL_PAD for c in range(ncols)]
    minw = [max(m, min(em_width("".join(paras[0][c])), 8) * TABLE_EM + CELL_PAD) for c, m in enumerate(minw)]
    weight = [sum(em_width("".join(r[c])) for r in body) / max(len(body), 1) + 1 for c in range(ncols)]
    widths = [int(x) for x in allocate(need, minw, weight, TEXT_W)]
    widths[widths.index(max(widths))] += TEXT_W - sum(widths)
    for g, x in zip(grid, widths):
        g.set(w("w"), str(x))
    tblPr = tbl.find(w("tblPr"))
    put(tblPr, "tblW", TBLPR_ORDER, {"w": TEXT_W, "type": "dxa"})
    put(tblPr, "tblLayout", TBLPR_ORDER, {"type": "fixed"})

    # 数字列居中（第一列是行名，始终左对齐）
    numeric = [c > 0 and sum(bool(NUMERIC.match("".join(r[c]))) for r in body) >= 0.7 * len(body)
               and max(em_width("".join(r[c])) for r in body) <= 14 for c in range(ncols)]

    # 估算表高：小表整表不跨页，大表只保证表头不孤悬页底
    lines = [max(max(1, math.ceil(em_width(t) * TABLE_EM / max(widths[c] - CELL_PAD, 1))) for c in range(ncols) for t in r[c]) for r in paras]
    height_pt = sum(n * 13 + 4 for n in lines)
    keep_rows = len(rows) - 1 if height_pt < 330 else min(2, len(rows) - 1)

    for i, (tr, r) in enumerate(zip(rows, cells)):
        put(first_child(tr, "trPr"), "cantSplit", ["cantSplit"])
        for c, tc in enumerate(r):
            put(first_child(tc, "tcPr"), "tcW", TCPR_ORDER, {"w": widths[c], "type": "dxa"})
            for p in tc.iter(w("p")):
                set_spacing(p, before=24, after=24, line=264)
                put(ppr(p), "jc", PPR_ORDER, {"val": "center" if numeric[c] else "left"})
                if i < keep_rows:
                    put(ppr(p), "keepNext", PPR_ORDER)


def number_captions(body):
    n = 0
    for p in body.iter(w("p")):
        if style_of(p) == "ImageCaption":
            n += 1
            r = ET.Element(w("r"))
            rpr = ET.SubElement(r, w("rPr"))
            ET.SubElement(rpr, w("b"))
            ET.SubElement(rpr, w("bCs"))
            t = ET.SubElement(r, w("t"), {XML_SPACE: "preserve"})
            t.text = f"图 {n}　"
            p.insert(1, r)
    return n


def left_align_long_code(body):
    """正文里有很长的行内代码（路径）时改为左对齐，否则两端对齐会把这一行的字距拉得很开。"""
    for p in body.findall(w("p")):
        for r in p.findall(w("r")):
            rs = r.find(f"{w('rPr')}/{w('rStyle')}")
            if rs is not None and rs.get(w("val")) == "VerbatimChar" and len(text_of(r)) >= 18:
                put(ppr(p), "jc", PPR_ORDER, {"val": "left"})
                break


def space_around_tables(body):
    kids = list(body)
    for i, el in enumerate(kids):
        if local(el) != "tbl":
            continue
        prev = kids[i - 1] if i > 0 else None
        nxt = kids[i + 1] if i + 1 < len(kids) else None
        if prev is not None and local(prev) == "p" and not style_of(prev).startswith("Heading"):
            set_spacing(prev, after=120)
            if text_of(prev).rstrip().endswith(("：", ":")):
                put(ppr(prev), "keepNext", PPR_ORDER)
        if nxt is not None and local(nxt) == "p" and not style_of(nxt).startswith("Heading"):
            set_spacing(nxt, before=180)


def build_cover(body):
    """封面：标题、副标题、姓名栏、“关于本报告的说明”；目录挪到说明之后（目录标题样式自带分页）。"""
    kids = list(body)
    sdt = body.find(w("sdt"))
    info = next(el for el in kids if local(el) == "p" and text_of(el).startswith("姓名"))
    note = kids[kids.index(info) + 1]
    assert local(note) == "tbl" and text_of(note).startswith("关于本报告的说明")
    body.remove(sdt)
    body.insert(list(body).index(note) + 1, sdt)
    put(ppr(info), "jc", PPR_ORDER, {"val": "center"})
    set_spacing(info, before=900, after=1500)
    for r in info.findall(w("r")):
        rpr = first_child(r, "rPr")
        put(rpr, "sz", RPR_ORDER, {"val": 24})
        put(rpr, "szCs", RPR_ORDER, {"val": 24})


def run_xml(text):
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def field_xml(code):
    return (f'<w:r><w:fldChar w:fldCharType="begin" /></w:r><w:r><w:instrText xml:space="preserve"> {code} </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate" /></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end" /></w:r>')


def add_header_footer(files, sect, title, subtitle):
    """页眉：左标题右副标题；页脚：第 X 页 / 共 Y 页。封面不显示（titlePg）。"""
    head = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:{{tag}} xmlns:w="{W_NS}" xmlns:r="{R_NS}">'
    files["word/header1.xml"] = (head.format(tag="hdr") + '<w:p><w:pPr><w:pStyle w:val="Header" /></w:pPr>'
                                 + run_xml(title) + "<w:r><w:tab /></w:r>" + run_xml(subtitle) + "</w:p></w:hdr>").encode("utf-8")
    files["word/footer1.xml"] = (head.format(tag="ftr") + '<w:p><w:pPr><w:pStyle w:val="Footer" /></w:pPr>'
                                 + run_xml("第 ") + field_xml("PAGE") + run_xml(" 页 / 共 ") + field_xml("NUMPAGES") + run_xml(" 页")
                                 + "</w:p></w:ftr>").encode("utf-8")
    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    rels = rels.replace("</Relationships>",
                        f'<Relationship Id="rIdHdr1" Type="{R_NS}/header" Target="header1.xml" />'
                        f'<Relationship Id="rIdFtr1" Type="{R_NS}/footer" Target="footer1.xml" /></Relationships>')
    files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    ct = files["[Content_Types].xml"].decode("utf-8")
    ct = ct.replace("</Types>",
                    '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml" />'
                    '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml" /></Types>')
    files["[Content_Types].xml"] = ct.encode("utf-8")
    for tag, rid in (("footerReference", "rIdFtr1"), ("headerReference", "rIdHdr1")):
        sect.insert(0, ET.Element(w(tag), {w("type"): "default", f"{{{R_NS}}}id": rid}))
    put(sect, "titlePg", SECT_ORDER)


def postprocess(path, title, subtitle):
    with zipfile.ZipFile(path) as zin:
        files = {n: zin.read(n) for n in zin.namelist()}
    xml = files["word/document.xml"].decode("utf-8")
    for prefix, uri in re.findall(r'xmlns:(\w+)="([^"]+)"', xml[:3000]):
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(xml)
    body = root.find(w("body"))

    n_callout = n_table = 0
    for tbl in body.iter(w("tbl")):
        first_row = tbl.find(w("tr"))
        is_header = first_row.find(f"{w('trPr')}/{w('tblHeader')}") is not None
        if len(tbl.find(w("tblGrid"))) == 1 and not is_header:
            format_callout(tbl)
            n_callout += 1
        else:
            format_table(tbl)
            n_table += 1
    build_cover(body)
    space_around_tables(body)
    left_align_long_code(body)
    n_fig = number_captions(body)
    add_header_footer(files, body.find(w("sectPr")), title, subtitle)

    st = files["word/settings.xml"].decode("utf-8")
    if "<w:compat>" not in st:
        st = st.replace("<w:rsids", '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15" /></w:compat><w:rsids', 1)
        files["word/settings.xml"] = st.encode("utf-8")

    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, xml_declaration=True, encoding="UTF-8")
    files["word/document.xml"] = buf.getvalue()
    write_zip(path, files)
    print(f"排版后处理：{n_table} 张表、{n_callout} 个提示框、{n_fig} 张图")


# ---------- 5. 目录 ----------
WORD_APP = Path("/Applications/Microsoft Word.app")


def refresh_toc_with_word(path):
    """用 Word 生成目录（含页码）后另存。在临时目录里做，避免沙盒权限弹窗。成功返回 True。"""
    if platform.system() != "Darwin" or not WORD_APP.exists():
        return False
    work = Path(tempfile.mkdtemp())
    tmp = work / "report.docx"
    shutil.copy(path, tmp)
    script = f'''
    tell application "Microsoft Word"
        open (POSIX file "{tmp}")
        set d to active document
        update (table of contents 1 of d)
        save d
        close d saving no
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print("Word 刷新目录失败，改为打开时提示更新：", getattr(e, "stderr", e))
        return False
    shutil.copy(tmp, path)
    shutil.rmtree(work)
    return True


def request_field_update(path):
    """让 Word 打开时询问是否更新域，目录才会自动生成。按 OOXML 顺序插在 <w:rsids> 之前。"""
    with zipfile.ZipFile(path) as zin:
        files = {n: zin.read(n) for n in zin.namelist()}
    st = files["word/settings.xml"].decode("utf-8")
    if "updateFields" not in st:
        anchor = "<w:compat>" if "<w:compat>" in st else "<w:rsids"          # OOXML 顺序：updateFields 在 compat、rsids 之前
        assert anchor in st
        st = st.replace(anchor, '<w:updateFields w:val="true" />' + anchor, 1)
        files["word/settings.xml"] = st.encode("utf-8")
        write_zip(path, files)


# ---------- 3. 组装 ----------
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    draw_mainline(FIG / "mainline.png")
    draw_framework(FIG / "framework.png")
    draw_tree(FIG / "tree_depth2.png")

    md = MD.read_text(encoding="utf-8")
    md, n1 = re.subn(r"```\n第一篇 问题：单棵树.*?\n```", "![四篇分析主线（下方小字为各篇的关键数字）](outputs/report/mainline.png)", md, flags=re.S)
    md, n2 = re.subn(r"```\n\s+集成学习（把多个模型的结果组合起来）.*?\n```", "![集成学习框架图](outputs/report/framework.png)", md, flags=re.S)
    md, n3 = re.subn(r"```\nMode ≤ 106 ─┬─.*?\n```", "![深度 2 的决策树（叶节点里是病理 / 其余片段数）](outputs/report/tree_depth2.png)", md, flags=re.S)
    assert n1 == 1 and n2 == 1 and n3 == 1, (n1, n2, n3)
    md = re.sub(r"(!\[[^\]]*\]\([^)]+\))(?!\{)", r"\1{width=100%}", md)
    md = re.sub(r"(?m)^---\n", "", md)                                    # 有了标题层级，Word 里不需要分隔线
    title_line = md.splitlines()[0]
    assert title_line.startswith("# ")
    title, subtitle = [t.strip() for t in title_line[2:].split("——")]
    md = md.split("\n", 1)[1]

    work = Path(tempfile.mkdtemp())
    (work / "report.md").write_text(md, encoding="utf-8")
    (work / "callout.lua").write_text(CALLOUT_LUA, encoding="utf-8")
    build_reference(work / "reference.docx")
    proc = subprocess.run(["pandoc", str(work / "report.md"), "-f", "markdown+lists_without_preceding_blankline", "-o", str(DOCX),
                           "--reference-doc", str(work / "reference.docx"), "--resource-path", str(BASE),
                           "--lua-filter", str(work / "callout.lua"), "--syntax-highlighting=none",
                           "--shift-heading-level-by=-1", "--toc", "--toc-depth=2", "-M", "toc-title=目录", "-M", "lang=zh-CN",
                           "-M", f"title={title}", "-M", f"subtitle={subtitle}"],
                          stderr=subprocess.PIPE, text=True)                  # 只在出错时显示 pandoc 的输出（平时全是缺翻译的警告）
    if proc.returncode:
        raise SystemExit(proc.stderr)
    shutil.rmtree(work)
    postprocess(DOCX, title, subtitle)
    if refresh_toc_with_word(DOCX):
        print("已用 Microsoft Word 生成目录和页码")
    else:
        request_field_update(DOCX)
    print("Wrote", DOCX)


if __name__ == "__main__":
    main()
