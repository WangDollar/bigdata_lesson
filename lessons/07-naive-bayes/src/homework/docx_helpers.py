"""python-docx 小工具：中文字体、标题、表格、图片、页码。两份文档共用。"""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

LATIN, SONG, HEI, KAI = 'Times New Roman', '宋体', '黑体', '楷体'


def _fonts(rpr_owner, latin, east):
    rpr = rpr_owner.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    for attr in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        rfonts.attrib.pop(qn(attr), None)
    rfonts.set(qn('w:ascii'), latin)
    rfonts.set(qn('w:hAnsi'), latin)
    rfonts.set(qn('w:eastAsia'), east)


class Doc:
    def __init__(self, body_pt=12, line=1.3):
        self.d = Document()
        self.body_pt = body_pt
        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.top_margin = sec.bottom_margin = Cm(2.54)
        sec.left_margin = sec.right_margin = Cm(2.8)
        normal = self.d.styles['Normal']
        normal.font.size = Pt(body_pt)
        _fonts(normal.element, LATIN, SONG)
        normal.paragraph_format.line_spacing = line
        normal.paragraph_format.space_after = Pt(3)
        for name, size, before, after in [('Title', 18, 0, 6), ('Heading 1', 15, 12, 6),
                                          ('Heading 2', 13, 8, 4), ('Heading 3', 12, 6, 3)]:
            st = self.d.styles[name]
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.italic = False
            st.font.color.rgb = RGBColor(0, 0, 0)
            _fonts(st.element, LATIN, HEI)
            pf = st.paragraph_format
            pf.space_before, pf.space_after = Pt(before), Pt(after)
            pf.line_spacing = 1.2
            pf.keep_with_next = True
            # Title 默认有下边框，去掉
            ppr = st.element.get_or_add_pPr()
            for b in ppr.findall(qn('w:pBdr')):
                ppr.remove(b)
        self._footer_page_number()

    def _footer_page_number(self):
        p = self.d.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for tag, text in [('begin', None), (None, 'PAGE'), ('end', None)]:
            if tag:
                el = OxmlElement('w:fldChar')
                el.set(qn('w:fldCharType'), tag)
            else:
                el = OxmlElement('w:instrText')
                el.set(qn('xml:space'), 'preserve')
                el.text = text
            run._r.append(el)
        run.font.size = Pt(9)

    # ---------- 文本 ----------
    def _runs(self, p, text, size=None, font=None, color=None, italic=False):
        """'**' 切换加粗。"""
        bold = False
        for i, seg in enumerate(text.split('**')):
            if i:
                bold = not bold
            if not seg:
                continue
            r = p.add_run(seg)
            r.bold = bold
            r.italic = italic
            if size:
                r.font.size = Pt(size)
            if font:
                _fonts(r._r, LATIN, font)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)
        return p

    def title(self, text, sub=None):
        p = self.d.add_paragraph(style='Title')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._runs(p, text)
        if sub:
            q = self.d.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self._runs(q, sub)
            q.paragraph_format.space_after = Pt(10)

    def h1(self, text):
        self._runs(self.d.add_paragraph(style='Heading 1'), text)

    def h2(self, text):
        self._runs(self.d.add_paragraph(style='Heading 2'), text)

    def h3(self, text):
        self._runs(self.d.add_paragraph(style='Heading 3'), text)

    def p(self, text, indent=True, size=None, font=None, color=None, align=None, after=None,
          left=None, italic=False, line=None):
        para = self.d.add_paragraph()
        if line:
            para.paragraph_format.line_spacing = line
        if indent:
            para.paragraph_format.first_line_indent = Pt((size or self.body_pt) * 2)
        if left is not None:
            para.paragraph_format.left_indent = Cm(left)
        if align == 'center':
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        return self._runs(para, text, size=size, font=font, color=color, italic=italic)

    def f(self, text, size=None):
        """公式或填空行：不首行缩进，整体左缩进。"""
        return self.p(text, indent=False, left=0.74, size=size)

    def label(self, head, text):
        """“训练时：……” 这类以短标签开头的一段，标签加粗，悬挂对齐。"""
        para = self.d.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.74)
        self._runs(para, f'**{head}**{text}')
        return para

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------- 表格 ----------
    def table(self, rows, widths_cm, header=True, size=10.5, align=None, shade='E7E6E6'):
        if self.d.paragraphs:  # 表前的引导句和表放在同一页
            self.d.paragraphs[-1].paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = 'Table Grid'
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(rows):
            trpr = t.rows[i]._tr.get_or_add_trPr()
            trpr.append(OxmlElement('w:cantSplit'))
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths_cm[j])
                para = cell.paragraphs[0]
                para.paragraph_format.first_line_indent = Pt(0)
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.15
                a = (align[j] if align else 'center')
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER if a == 'center' else WD_ALIGN_PARAGRAPH.LEFT
                text = str(val)
                if header and i == 0:
                    text = f'**{text}**'
                    tcpr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement('w:shd')
                    shd.set(qn('w:val'), 'clear')
                    shd.set(qn('w:color'), 'auto')
                    shd.set(qn('w:fill'), shade)
                    tcpr.append(shd)
                lines = text.split('\n')
                self._runs(para, lines[0], size=size)
                for extra in lines[1:]:
                    q = cell.add_paragraph()
                    q.alignment = para.alignment
                    q.paragraph_format.space_after = Pt(0)
                    q.paragraph_format.first_line_indent = Pt(0)
                    self._runs(q, extra, size=size)
                if i < len(rows) - 1:  # 整张表不跨页
                    for para_ in cell.paragraphs:
                        para_.paragraph_format.keep_with_next = True
        # 表格后留一点空
        gap = self.d.add_paragraph()
        gap.paragraph_format.space_after = Pt(2)
        gap.paragraph_format.line_spacing = 0.6
        return t

    def caption(self, text, size=10.5):
        return self.p(text, indent=False, align='center', size=size, after=8)

    def image(self, path, width_cm, caption=None):
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        if caption:
            self.caption(caption)

    def save(self, path):
        self.d.save(str(path))
