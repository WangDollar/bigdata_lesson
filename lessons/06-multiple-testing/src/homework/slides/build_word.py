"""翻转课堂讲解提纲（Word）+ 手持 A4 讲稿。

不重跑任何分析：数字全部读自 outputs/homework/results.json，两张图复用 build_slides.py 的画图函数。
用法：
    /opt/homebrew/Caskroom/miniconda/base/envs/mtest/bin/python build_word.py
输出：讲解提纲.docx、讲稿_A4.pdf（由 讲稿.html 打印）
"""
import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from build_slides import FIG, HERE, OUT, fig_perm, fig_pvalues, to_pdf

SONG, HEI, LATIN = "宋体", "黑体", "Times New Roman"
BODY, SMALL = 12, 10.5


def set_font(obj, east=SONG, size=None, bold=None):
    obj.font.name = LATIN
    if size:
        obj.font.size = Pt(size)
    if bold is not None:
        obj.font.bold = bold
    rfonts = obj._element.get_or_add_rPr().get_or_add_rFonts()
    for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(theme), None)
    rfonts.set(qn("w:eastAsia"), east)


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.4)

    # 文档默认字体去掉主题字体，东亚语言设为中文，避免表格等处回落到 Calibri / 等线
    rpr = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
    rfonts = rpr.find(qn("w:rFonts"))
    for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(theme), None)
    for attr, val in (("w:ascii", LATIN), ("w:hAnsi", LATIN), ("w:eastAsia", SONG)):
        rfonts.set(qn(attr), val)
    rpr.find(qn("w:lang")).set(qn("w:eastAsia"), "zh-CN")

    normal = doc.styles["Normal"]
    set_font(normal, size=BODY)
    normal.paragraph_format.line_spacing = 1.3
    normal.paragraph_format.space_after = Pt(3)
    for name, size, before in [("Heading 1", 16, 16), ("Heading 2", 13.5, 10)]:
        st = doc.styles[name]
        set_font(st, east=HEI, size=size, bold=True)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.italic = False
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(5)
        linked = doc.styles[f"{name} Char"]
        set_font(linked, east=HEI)
        linked.font.color.rgb = RGBColor(0, 0, 0)

    # 页脚页码
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r, t = OxmlElement("w:r"), OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)


def runs(p, text, size=None):
    """**粗体** 标记转成粗体 run。"""
    for i, part in enumerate(text.split("**")):
        if part:
            r = p.add_run(part)
            r.bold = i % 2 == 1
            if size:
                r.font.size = Pt(size)
    return p


def para(doc, text, size=None):
    return runs(doc.add_paragraph(), text, size)


def bullets(doc, items, level=1):
    for item in items:
        runs(doc.add_paragraph(style="List Bullet" if level == 1 else "List Bullet 2"), item)


def shade(cell, fill="E7E6E6"):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(doc, rows, widths, center_from=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            if center_from is not None and j >= center_from:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            runs(p, f"**{text}**" if i == 0 else text, SMALL)
            if i == 0:
                shade(cell)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(doc, path, caption):
    doc.add_picture(str(path), width=Cm(13.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = runs(doc.add_paragraph(), caption, SMALL)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def dec(v):
    """三位小数；除不尽时保留四位（如 0.0225）。"""
    return f"{v:.3f}" if abs(v * 1000 - round(v * 1000)) < 1e-9 else f"{v:.4f}"


def n(x):
    return f"{x:,.0f}".replace("-", "−")


def build(res, path):
    simA, simB = res["simA"], res["simB"]["signal_1.0"]
    c1, c2, c4 = res["C1"], res["C2"], res["C4"]
    a1, a2 = res["adv1_permutation"], res["adv2_storey"]
    m, exp0 = c1["m"], c1["expected_raw_if_all_null"]
    raw, bonf, bh = (c1["counts"][k] for k in ("raw", "bonferroni", "bh"))
    ind = a1["independent_reference"]
    u = next(r for r in c4["rows"] if r["probe"] == "U22376_cds2_s_at")
    failed = [r["probe"] for r in c4["rows"] if not r["replicated"]]
    pi0, lo, hi = a2["pi0"], *a2["pi0_range_lambda_0.2_0.8"]
    flat = m * 0.05
    strat = a1["share_any_by_overlap"]
    toy = res["toy_example"]

    doc = Document()
    setup(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(f"{n(m)} 个探针里，有多少差异是真的？"), east=HEI, size=18, bold=True)
    p = para(doc, "第五课 多重检验校正｜翻转课堂讲解提纲")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(doc, f"文中数字均来自我运行 analysis.py 的结果（随机种子 {res['protocol']['seed']}）", SMALL)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1 题目情景
    doc.add_heading("1　题目情景", level=1)
    sp = res["data"]["split"]
    bullets(doc, [
        f"**数据：**Golub 白血病芯片数据，72 个样本 × {n(m)} 个探针，表达值是芯片原始强度。",
        f"**划分：**train {sp['train']['ALL'] + sp['train']['AML']} 例（ALL {sp['train']['ALL']} / AML {sp['train']['AML']}）用来发现；"
        f"test {sp['test']['ALL'] + sp['test']['AML']} 例（ALL {sp['test']['ALL']} / AML {sp['test']['AML']}）用来验证，锁定候选之前不打开。",
        "**我想回答两个问题：**",
    ])
    bullets(doc, ["**① 数量：**AML 和 ALL 之间，大概有多少探针真的有表达差异？",
                  "**② 点名：**能可靠地点名哪几个，值得进一步验证？"], level=2)
    bullets(doc, [
        f"**为什么是多重检验问题：**每个探针做一次检验，一共 {n(m)} 次。即使全部没有差异，"
        f"按 p ≤ 0.05 也会出现约 {n(m)} × 0.05 ≈ {n(exp0)} 个“显著”。",
        f"**看结果之前定好的做法：**每个探针做双侧 Welch t 检验，方向记为 AML − ALL；α = 0.05；"
        f"模拟和置换各重复 {n(res['protocol']['repeats'])} 次。",
    ])

    # 2 基础问题
    doc.add_heading("2　基础问题：上节课的几个概念", level=1)
    doc.add_heading("2.1　这几个概念各管什么，又不是什么？", level=2)
    table(doc, [
        ["概念", "是什么", "不是什么"],
        ["p 值", "假设“没有差异”成立时，看到至少这么极端结果的概率", "不是“没有差异”为真的概率"],
        ["家族 m", f"实际做过的全部检验，本题 m = {n(m)}", "不是最后报告的那几个"],
        ["FWER", "至少出现 1 个假发现的概率；Bonferroni 把阈值改为 α / m 来控制它", "不是“每个检验各错 5%”"],
        ["FDR", "所有发现里，假发现所占比例的长期平均；BH 控制它", "不保证这一批里每个都是真的"],
        ["校正后 p 值", "在给定家族和方法下，刚好能被拒绝的整体水平", "不是探针本身的属性：换了家族就会变"],
    ], [2.6, 7.4, 6.2])
    para(doc, "一句话：**FWER 管“有没有错”，FDR 管“错了多少”。**")

    doc.add_heading("2.2　同一组 p 值，两种校正为什么结果不同？", level=2)
    para(doc, "5 个教学用 p 值（不是真实探针），m = 5，α = 0.05：")
    k_bonf = sum(v <= 0.05 for v in toy["bonferroni"])
    k_bh = sum(v <= 0.05 for v in toy["bh"])
    rows = [["排名 i", "p 值", "Bonferroni 门槛 0.05/5", "BH 门槛 i × 0.05/5"]]
    for i, pv in enumerate(toy["raw"], 1):
        bh_thr = i * 0.01
        rows.append([str(i), f"{pv:.3f}", f"0.01　{'✓' if pv <= 0.01 else '✗'}",
                     f"{bh_thr:.2f}　{'✓' if i <= k_bh else '✗'}" + ("（恰好压线）" if pv == bh_thr else "")])
    table(doc, rows, [2.2, 2.6, 5.0, 5.4], center_from=0)
    bullets(doc, [
        f"**Bonferroni：**每个检验用同一个门槛 0.01 → 发现 {k_bonf} 个。",
        f"**BH：**门槛随排名放宽，找到最后一个过线的位置 k = {k_bh}，前 {k_bh} 个都算发现 → 发现 {k_bh} 个。",
        "换成校正后的 p 值也一样：Bonferroni 为 " + "、".join(map(dec, toy["bonferroni"]))
        + "；BH 为 " + "、".join(map(dec, toy["bh"])) + f"。≤ 0.05 的分别是 {k_bonf} 个和 {k_bh} 个。",
    ])

    doc.add_heading("2.3　模拟 A：全部没有差异时，会“显著”多少个？", level=2)
    va, sa = simA["variation"], simA["summary"]
    para(doc, f"1,000 个变量全部没有差异，每次按 p ≤ 0.05 判断，重复 {n(sa['raw']['repeats'])} 次：")
    table(doc, [
        ["", "不校正", "Bonferroni", "BH"],
        ["平均“显著”个数", f"{va['raw']['mean_R']:.1f}", f"{va['bonferroni']['mean_R']:.3f}", f"{va['bh']['mean_R']:.3f}"],
        ["至少错 1 个的概率（FWER）", f"{sa['raw']['FWER_estimate']:.3f}", f"{sa['bonferroni']['FWER_estimate']:.3f}",
         f"{sa['bh']['FWER_estimate']:.3f}"],
    ], [6.0, 3.2, 3.2, 3.2], center_from=1)
    bullets(doc, [
        f"不校正时平均 {va['raw']['mean_R']:.1f} 个“显著”（理论 1,000 × 0.05 = 50），全部是假阳性；"
        f"90% 的实验落在 {va['raw']['p05_R']:.0f}–{va['raw']['p95_R']:.0f} 个，我第一次运行是 {simA['first_run_R']['raw']} 个。",
        "**我的理解：**50 是期望。期望可以逐项相加，所以“平均约 50 个”不需要检验相互独立；"
        "但每次具体有多少（波动大小）会受相关性影响，§4.1 用真实数据检查这一点。",
    ])

    doc.add_heading("2.4　模拟 B：加入真信号后，两种校正各付出什么代价？", level=2)
    para(doc, "1,000 个变量中 100 个有真实差异（效应 δ = 1.0），900 个没有；重复 1,000 次取平均：")
    table(doc, [
        ["", "不校正", "Bonferroni", "BH"],
        ["检出力（真信号被找到的比例）", f"{simB['raw']['mean_power']:.2f}", f"{simB['bonferroni']['mean_power']:.2f}",
         f"{simB['bh']['mean_power']:.2f}"],
        ["FDR（发现里假的比例）", f"{simB['raw']['FDR_estimate']:.2f}", f"{simB['bonferroni']['FDR_estimate']:.3f}",
         f"{simB['bh']['FDR_estimate']:.3f}"],
        ["FWER（至少错 1 个）", f"{simB['raw']['FWER_estimate']:.2f}", f"{simB['bonferroni']['FWER_estimate']:.3f}",
         f"{simB['bh']['FWER_estimate']:.3f}"],
    ], [6.0, 3.2, 3.2, 3.2], center_from=1)
    bullets(doc, [
        f"Bonferroni 几乎不犯错，但只找到 {simB['bonferroni']['mean_power']:.0%} 的真信号；"
        f"BH 找到 {simB['bh']['mean_power']:.0%}，错的比例仍在 5% 以内。",
        f"BH 的 FWER {simB['bh']['FWER_estimate']:.3f} 和 FDR {simB['bh']['FDR_estimate']:.3f} **不矛盾**："
        f"平均 {simB['bh']['mean_R']:.0f} 个发现里约 {simB['bh']['mean_V']:.1f} 个是假的，所以“至少错一个”很常见，但错的比例很低。",
        "**怎么选：**探索性筛选怕漏 → 控制 FDR，用 BH；确认性结论怕错 → 控制 FWER，用 Bonferroni。",
        f"BH 的实际 FDR（{simB['bh']['FDR_estimate']:.3f}）约等于 π0 × α = 0.9 × 0.05 = 0.045，"
        "π0 是零假设所占的比例。§4.2 会在真实数据上估计 π0。",
    ])

    # 3 主线问题
    doc.add_heading("3　主线问题：真实数据，证据逐步收紧", level=1)
    doc.add_heading(f"3.1　发现：对全部 {n(m)} 个探针做检验", level=2)
    para(doc, f"在 train 中对每个探针做一次 Welch t 检验，然后在整个家族（m = {n(m)}）上校正：")
    d = c1["n_direction_among_bh"]
    table(doc, [
        ["步骤", "个数", "含义"],
        ["全部探针", n(m), "检验家族 m"],
        ["raw p ≤ 0.05", n(raw), f"不校正；全部无差异时也会有约 {n(exp0)} 个"],
        ["BH ≤ 0.05", n(bh), f"控制 FDR 5%；其中 AML 偏低 {d['AML_lower']} 个、偏高 {d['AML_higher']} 个"],
        ["Bonferroni ≤ 0.05", n(bonf), f"控制 FWER 5%；raw p 门槛 0.05 / {n(m)} ≈ 7×10⁻⁶"],
        ["锁定候选", str(c4["n_locked"]), "BH 名单里 raw p 最小的 10 个，打开 test 前存档"],
        ["验证通过", str(c4["n_replicated"]), "在 test 中复现"],
    ], [3.6, 2.0, 10.6], center_from=None)
    ex = c2["example"]
    shrink = ex["full_bonferroni"] / ex["wrong_bonferroni"]
    bullets(doc, [
        f"**家族必须是全部 {n(m)} 个：**如果先挑出 p 值最小的 20 个、只在这 20 个里校正，"
        f"U22376_cds2_s_at 的校正 p 会从 5.98×10⁻⁵ 变成 1.68×10⁻⁷，小了 {shrink:.0f} 倍。"
        "这次 20 个在两种做法下都显著，数量没变，但“控制错误率”的保证已经不成立。",
    ])

    doc.add_heading("3.2　锁定候选，再打开 test", level=2)
    bullets(doc, [
        f"打开 test 之前，从 BH 名单里取 raw p 最小的 10 个存档锁定（{res['C3']['locked_at']}），同时写好判据：",
    ])
    bullets(doc, ["在 10 个候选组成的家族里 Bonferroni ≤ 0.05，即 test 中 raw p ≤ 0.005；",
                  "效应方向与 train 一致。"], level=2)
    bullets(doc, [
        f"**结果：**{c4['n_same_direction']} 个方向全部一致，**{c4['n_replicated']} 个通过**。",
        "**为什么先锁定：**如果看了 test 再挑，test 就成了发现的一部分，不能再当验证。",
    ])

    doc.add_heading("3.3　通过验证，效应能照搬吗？", level=2)
    bullets(doc, [
        f"**例子 U22376_cds2_s_at：**train 中 AML − ALL = {n(u['discovery_delta'])}"
        f"（95% CI {n(u['disc_low'])} 到 {n(u['disc_high'])}），test 中 {n(u['delta_AML_minus_ALL'])}"
        f"（{n(u['ci95_low'])} 到 {n(u['ci95_high'])}），只剩 {u['ratio']:.0%}，但仍通过（候选家族 Bonferroni {u['bonferroni']:.3f}）。",
        f"10 个候选里 {c4['n_shrunk']} 个在 test 中效应变小，test / train 比值的中位数是 {c4['median_ratio']:.2f}。",
        "**赢家诅咒：**按 p 值挑出来的，往往是“真效应 + 恰好偏大的噪声”。所以引用效应大小时，用 test 中的数值。",
        f"**没通过的 3 个**（{'、'.join(failed)}）：方向都一致，test 中 raw p < 0.05，95% CI 不跨 0，只是没过 0.005。"
        "未通过 ≠ 没有效应，只能说按预定标准证据不足。",
        "**单位：**数据没有取 log，原始强度中 28.7% 是负值，所以只报强度上的均值差，不说“几倍”。",
    ])

    # 4 进阶问题
    doc.add_heading("4　进阶问题", level=1)
    doc.add_heading("4.1　进阶①　置换检验：探针相关时，“偶然显著”会波动多大？", level=2)
    bullets(doc, [
        f"§1 里的 {n(exp0)} 是按“检验相互独立”算的参照。但基因往往成组调控，探针之间是相关的。",
        f"**做法：**把 train 的 ALL/AML 标签随机打乱 {n(a1['n_perm'])} 次，每次重算 {n(m)} 个检验。"
        "打乱后标签与表达无关，但探针之间的相关性原样保留。",
    ])
    figure(doc, FIG / "s08-perm.png",
           f"图 1　打乱标签 {n(a1['n_perm'])} 次时 raw p ≤ 0.05 的探针数。灰带：假设检验相互独立时的 "
           f"{n(ind['mean'])} ± 2×{ind['sd']:.0f}；黑线：实际标签")
    table(doc, [
        ["", "raw p ≤ 0.05 的个数", "补充"],
        ["假设检验相互独立", f"{n(ind['mean'])} ± {ind['sd']:.0f}", "二项分布的平均 ± 标准差"],
        [f"打乱标签 {n(a1['n_perm'])} 次", f"{n(a1['raw']['mean'])} ± {a1['raw']['sd']:.0f}",
         f"5%–95%：{n(a1['raw']['p05'])}–{n(a1['raw']['p95'])}，最大 {n(a1['raw']['max'])}"],
        ["实际标签", n(a1["observed_raw"]), f"超过全部 {n(a1['n_perm'])} 次，置换 p = 1/{n(a1['n_perm'] + 1)}"],
    ], [4.4, 4.4, 7.4], center_from=None)
    bullets(doc, [
        f"平均值差不多（{n(a1['raw']['mean'])} 对 {n(ind['mean'])}）→ 印证“期望不需要独立”；",
        f"标准差大了约 {a1['raw']['sd'] / ind['sd']:.0f} 倍 → 探针相关让“偶然显著”的个数波动很大；",
        f"实际的 {n(a1['observed_raw'])} 超过全部打乱结果 → AML 与 ALL 之间确实存在大量真实差异；",
        f"但不能用“{n(raw)} − {n(exp0)} = {n(raw - exp0)}”来算有多少是真的：零假设下的个数本身就可能在 "
        f"{n(a1['raw']['p05'])}–{n(a1['raw']['p95'])} 之间波动。",
    ])

    doc.add_heading("4.2　进阶②　Storey π0：大概有多少探针真有差异？", level=2)
    bullets(doc, [
        "**思路：**真零假设的 p 值服从 0–1 均匀分布。p > 0.5 的部分几乎都来自零假设，"
        "所以直方图右侧“平坦部分”的高度反映零假设所占的比例 π0。",
    ])
    figure(doc, FIG / "s09-pvalues.png",
           f"图 2　train 中 {n(m)} 个探针的原始 p 值分布。点线：全部无差异时每格应有 {flat:.0f} 个；"
           f"虚线：实际平坦部分每格约 {pi0 * flat:.0f} 个")
    est = a2["est_non_null"]
    bullets(doc, [
        f"**计算：**π0 = #{{p > 0.5}} ÷ ({n(m)} × 0.5) = {pi0:.2f}（相当于 {pi0 * flat:.0f} ÷ {flat:.0f}）"
        f"→ 非零约 {1 - pi0:.0%}，约 {round(est, -2):,.0f} 个探针。λ 在 0.2–0.8 之间变化时，π0 为 {lo:.2f}–{hi:.2f}，"
        f"对应约 {round(m * (1 - hi), -2):,.0f}–{round(m * (1 - lo), -2):,.0f} 个。",
        f"**但是：**打乱标签后再算 π0，中位数是 {a2['perm_pi0']['median']:.2f}，最低一次也到了 "
        f"{a2['perm_pi0']['min']:.3f}，和真实数据的 {pi0:.2f} 差不多 → 在探针相关的数据里，π0 的估计不稳定，"
        f"“约 {round(est, -2):,.0f}”只当数量级。",
        f"Storey q 值 = π0 × BH 校正后的 p 值。q ≤ 0.05 的探针有 {a2['n_q_le_0.05']} 个（BH 是 {a2['n_bh_le_0.05']} 个）。"
        "我不用它扩大名单：q 依赖不稳定的 π0，而且名单在打开 test 前已经锁定。",
    ])

    # 5 回答
    doc.add_heading("5　我的回答与边界", level=1)
    bullets(doc, [
        f"**问题① 有多少是真的？**差异以千计，约三分之一的探针。证据：实际显著数超过全部 {n(a1['n_perm'])} 次打乱结果；"
        f"Storey π0 ≈ {pi0:.2f}。但只当数量级。",
        f"**问题② 能可靠点名哪几个？**FDR 5% 下有 {n(bh)} 个候选；打开 test 前锁定 {c4['n_locked']} 个，"
        f"{c4['n_replicated']} 个在留出样本中复现。",
        "**边界：**",
    ])
    bullets(doc, [
        "test 与 train 来自同一研究、同一平台的原始划分 → 是留出验证，不是外部验证；",
        "探针 ≠ 基因，关联 ≠ 因果；BH 名单里还有 3 个 AFFX 质控探针，提示两组间可能有技术或批次差异；",
        "验证后效应普遍缩水 → 引用 test 中的效应；",
        "“约三分之一”依赖 π0 的估计，在相关数据里不稳定。",
    ], level=2)

    # 附 备查
    doc.add_heading("附　备查（被问到时再看）", level=1)
    doc.add_heading("附 A　打乱标签后，校正为什么仍有 7.5%–8.2% 出了“发现”？", level=2)
    para(doc, f"只有 11 个 AML。打乱后的“假 AML 组”里常会混入几个真 AML，随机期望是 11 × 11 / 38 ≈ "
              f"{a1['overlap_expected']:.2f} 个。混入人数偏离期望时，假分组和真分组仍然相关，真信号就漏了进来。")
    near, far = strat["overlap_2_4"], strat["overlap_0_1_or_5_7"]
    table(doc, [
        ["混入真 AML 的人数", "打乱次数", "至少 1 个发现：Bonferroni", "BH"],
        ["全部", n(a1["n_perm"]), f"{a1['share_any_bonferroni']:.1%}", f"{a1['share_any_bh']:.1%}"],
        ["2–4 个（接近期望）", n(near["n"]), f"{near['bonferroni']:.1%}", f"{near['bh']:.1%}"],
        ["0–1 或 5–7 个", n(far["n"]), f"{far['bonferroni']:.1%}", f"{far['bh']:.1%}"],
    ], [5.2, 2.8, 5.0, 3.2], center_from=1)
    para(doc, "接近期望时比例接近 5%，说明多出来的部分主要来自“打乱不充分”，而不是校正方法失效。")

    doc.add_heading("附 B　方法选择与复现", level=2)
    bullets(doc, [
        "**发现用 BH、验证用 Bonferroni：**筛选阶段怕漏，控制 FDR；确认阶段怕错，控制 FWER。验证家族只有 10 个，门槛 0.005，代价小。",
        "**Welch t 检验：**两组方差不一定相同，样本量也不平衡（11 对 27），Welch 不要求等方差。",
        "**不取 log、不说“倍数”：**原始强度中 28.7% 为负值，取不了 log，比值也没有意义；只报强度上的均值差和 95% CI。",
        "**需要检验相互独立吗：**期望 E[R] = m·α 不需要；Bonferroni 不需要（并集不等式）；"
        "BH 需要独立或正相关（PRDS），任意相关时可改用 BY。",
        f"**复现：**随机种子 {res['protocol']['seed']} 在运行前固定；模拟和置换各 1,000 次；"
        "真实数据部分（发现、锁定、验证）不涉及随机数。",
    ])

    doc.save(path)
    print("已导出", path)


def main():
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    fig_perm(res)
    fig_pvalues(res)
    build(res, HERE / "讲解提纲.docx")
    to_pdf(HERE / "讲稿.html", HERE / "讲稿_A4.pdf")


if __name__ == "__main__":
    main()
