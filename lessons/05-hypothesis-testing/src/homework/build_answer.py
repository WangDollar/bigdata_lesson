"""Build a student answer in DOCX and Markdown from the computed results."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

LESSON = Path(__file__).resolve().parents[2]  # lessons/05-hypothesis-testing
OUT = LESSON / 'report'
R = json.loads((LESSON/'outputs'/'homework'/'results.json').read_text())
P, A = R['primary'], R['adjusted']
G0, G1 = R['group_summary']
B, SD, MP = R['balance'], R['site_disease'], R['missing_profile']
EV, IX, PW, MW = R['exclude_va'], R['interaction'], R['power_examples'], R['mann_whitney']
MS = {m['site']: m for m in R['missingness_by_site']}
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(1.65)
sec.left_margin = sec.right_margin = Cm(1.8)
sec.header_distance = sec.footer_distance = Cm(.7)

SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
def f(x, nd=2):
    """Fixed decimals with a true minus sign, so negative values match the rest of the text."""
    return f'{x:.{nd}f}'.replace('-', '−')
def sci(x):
    mantissa, exponent = f'{x:.2e}'.split('e')
    return f'{mantissa}×10{str(int(exponent)).translate(SUP)}'
def ci(lo, hi, nd=2):
    return f'［{f(lo,nd)}，{f(hi,nd)}］'

def font_style(style, size, bold=False):
    s = doc.styles[style]
    s.font.name = 'Arial'
    s.font.size = Pt(size)
    s.font.bold = bold
    s.font.color.rgb = RGBColor(0,0,0)
    rp = s.element.get_or_add_rPr()
    rf = rp.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rp.insert(0, rf)
    for attr in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
        rf.attrib.pop(qn('w:'+attr), None)
    for attr in ('ascii','hAnsi','cs'): rf.set(qn('w:'+attr),'Arial')
    # DengXian ships with Windows 10+ and Office for Mac, so pages break the same on both.
    rf.set(qn('w:eastAsia'),'DengXian')
    cp = rp.find(qn('w:color'))
    if cp is not None:
        for attr in list(cp.attrib):
            if attr != qn('w:val'): del cp.attrib[attr]
    pp=s.paragraph_format
    pp.line_spacing=1.16
    pp.space_after=Pt(5)
    for border in s.element.findall('.//'+qn('w:pBdr')): border.getparent().remove(border)

font_style('Normal',10.5)
font_style('Title',20,True)
font_style('Subtitle',10)
font_style('Heading 1',14,True)
font_style('Heading 2',11.5,True)
doc.styles['Title'].paragraph_format.space_after = Pt(8)
for name in ('Heading 1','Heading 2'):
    doc.styles[name].paragraph_format.space_before=Pt(10)
    doc.styles[name].paragraph_format.space_after=Pt(6)
    doc.styles[name].paragraph_format.keep_with_next=True

md=[]

def p(text, bold_label=None, size=None, before=None):
    para=doc.add_paragraph()
    if bold_label and text.startswith(bold_label):
        para.add_run(bold_label).bold=True
        para.add_run(text[len(bold_label):])
        md.append(f'**{bold_label}**{text[len(bold_label):]}\n')
    else:
        para.add_run(text)
        md.append(text+'\n')
    if size:
        for r in para.runs: r.font.size=Pt(size)
    if before is not None: para.paragraph_format.space_before=Pt(before)
    para.paragraph_format.widow_control=True
    return para

def h(text, level=1, new_page=False):
    para=doc.add_paragraph(text,style=f'Heading {level}')
    if new_page: para.paragraph_format.page_break_before=True
    md.append('#'*(level+1)+' '+text+'\n')
    return para

def img(path, caption, width):
    para=doc.add_paragraph()
    para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.keep_with_next=True
    para.paragraph_format.space_after=Pt(2)
    para.add_run().add_picture(str(OUT/path), width=Cm(width))
    cap=doc.add_paragraph()
    cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=cap.add_run(caption); run.font.size=Pt(9)
    md.append(f'![{caption}]({path})\n')
    md.append(caption+'\n')

def table(headers, rows, widths):
    t=doc.add_table(rows=1,cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for col,width in zip(t.columns,widths): col.width=Cm(width)
    for i,header in enumerate(headers): t.rows[0].cells[i].text=header
    for row in rows:
        cells=t.add_row().cells
        for i, value in enumerate(row): cells[i].text=str(value)
    for rowi,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit'); trpr.append(cant)
        if rowi==0:
            repeat=OxmlElement('w:tblHeader'); trpr.append(repeat)
        for i,cell in enumerate(row.cells):
            cell.width=Cm(widths[i]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr()
            shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E7EDF3' if rowi==0 else 'FFFFFF'); pr.append(shade)
            borders=OxmlElement('w:tcBorders')
            for side in ('top','left','bottom','right'):
                b=OxmlElement('w:'+side); b.set(qn('w:val'),'single'); b.set(qn('w:sz'),'4'); b.set(qn('w:color'),'D9D9D9'); borders.append(b)
            pr.append(borders)
            margins=OxmlElement('w:tcMar')
            for side,value in (('top','75'),('bottom','75'),('left','95'),('right','95')):
                m=OxmlElement('w:'+side); m.set(qn('w:w'),value); m.set(qn('w:type'),'dxa'); margins.append(m)
            pr.append(margins)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(1)
                para.paragraph_format.space_before=Pt(1)
                para.paragraph_format.line_spacing=1.10
                for run in para.runs:
                    run.font.size=Pt(9.5)
                    run.font.bold=(rowi==0)
    md.append('| '+' | '.join(headers)+' |')
    md.append('| '+' | '.join('---' for _ in headers)+' |')
    md.extend('| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |' for row in rows)
    md.append('')
    return t

attenuation = 1-A['coefficient']/P['mean_difference']
cov = A['covariates']
site_gaps = [-cov[k]['coefficient'] for k in cov if k.startswith('C(site)')]
dup = R['duplicate_profile_groups']
zeros = R['implausible_zeros']
out0, out1 = R['tukey_outliers']['0'], R['tukey_outliers']['1']
fmt_list = lambda xs: '、'.join(f'{x:.0f}' for x in xs)

title='统计检验与假设设计学生答卷'
doc.add_paragraph(title,'Title'); md.append('# '+title+'\n')
p(f"这份作业我想回答一个问题：有心脏病证据的人，最大运动心率和没有的人一样吗？先把结论放在前面：疾病组平均低了约 {abs(P['mean_difference']):.1f} 次/分；把年龄、性别和中心考虑进去以后，差距缩小到约 {abs(A['coefficient']):.1f} 次/分，但仍然存在。不过这是观察性资料，只能说两者有关联，不能说心脏病“导致”心率降低。")
p('数据来源：hypothesis/data/heart-disease-hypothesis.csv；字段解释见同目录 README.md。',size=9)

h('任务 1：把问题变成假设')
p('题目里的“有关”太笼统，没法直接检验：均值不同、比例不同、分布不同，都可以叫“有关”。心率是连续变量，比较的又是两组人，所以我把“有关”落到最直观的一个量上，也就是两组平均心率的差 Δ。下面是我填的各项。')
table(['填写项','我的答案'],[
 ('研究问题','在这四个中心的资料里，有疾病证据和没有疾病证据的人，最大运动心率的总体平均值一样吗？'),
 ('研究对象','这四个中心的受试者。主要分析只用测到了心率的人，所以结论不能直接推广到所有人群，也不能推到没测心率的人身上。'),
 ('比较对象','disease=1（num>0，有疾病证据）对比 disease=0（num=0）。我没把后者叫“健康组”：num=0 只说明没达到这个标签的疾病标准，不等于完全健康。'),
 ('主要终点','thalach，也就是最大运动心率，单位次/分。看结果之前就定好了，而且只定这一个。'),
 ('分析单位','一行就是一名受试者。site 是数据来自哪个中心，不是随机分出来的组。'),
 ('H₀','设两组总体平均心率分别是 μ₁、μ₀，Δ=μ₁−μ₀（固定用疾病组减无疾病组）。H₀：Δ=0，也就是两组平均值一样。'),
 ('H₁','H₁：Δ≠0。Δ 是负数就说明疾病组更低。'),
 ('α','0.05，跑检验之前就定死。'),
 ('单侧或双侧及理由','双侧。题目问的是“是否不同”，不是“是否更低”，反方向的差异也有意义。而且方向必须在看数据前定；看到结果再改成单侧，等于看了答案再改题目。'),
 ('检验与纳入规则','用双侧 Welch 独立样本 t 检验。只排除分组或心率缺失的人；其他字段缺了不删，心率不填补，也不在看完结果后删“奇怪的点”。'),
], [3.4,14.0])
p('这些选择是我跑检验之前写下的。需要说明：这只是课堂练习的分析约定，不是正式的预注册。',size=9,before=4)

h('任务 2：错误类型与 power')
p('第一类错误：两组平均心率其实一样，我却拒绝了 H₀，说“疾病组心率不同”。放到这道题里，就是团队把一个不存在的关联写进了结论。只要检验假设成立、规则事先定好，犯这类错误的概率就是 α=0.05。',bold_label='第一类错误：')
p('第二类错误：两组其实不一样，我却没有拒绝 H₀，把真实的差异漏掉了。这个概率记作 β；power=1−β，也就是“差异真的存在时，我能把它查出来的概率”。',bold_label='第二类错误：')
p(f"影响 power 的因素：样本越多、真实差距越大、测量噪声越小，power 就越高。光这样说不够直观，我用本数据的标准差（约 {PW['planning_sd']:.1f} 次/分）当规划值算了一下（双侧 α=0.05，目标 power 80%）：想查出 10 次/分的差距，每组大约要 {PW['n_per_group_delta10']} 人；差距只有 5 次/分时，每组要 {PW['n_per_group_delta5']} 人，差距减半，样本要翻 4 倍；如果测量噪声让标准差翻倍，查出 10 次/分同样要每组 {PW['n_per_group_delta10_double_sd']} 人。",bold_label='影响 power 的因素：')
p('还要注意两点。第一，power 应该在设计研究时就算；拿这次观测到的差值去算“事后 power”，提供不了 p 值以外的新信息。第二，样本再大，也纠正不了混杂、选择偏倚或测量本身的系统偏差。')
p('未拒绝 H₀ 的边界：没拒绝 H₀ 只说明现有证据不够，不代表两组一样，更不等于“接受 H₀”。如果真想说明“差距小到可以忽略”，得事先定一个可以接受的差距范围，再做等效性检验。',bold_label='未拒绝 H₀ 的边界：')

h('任务 3：选择方法')
p('选方法时我不去死记“什么题配什么检验”，而是每次问自己三个问题：结局是连续变量还是分类变量？比较几组？数据是独立的还是配对的？这三个问题答完，方法基本就定了。下面五题只是选方法，真正做推断的只有预先定好的 thalach。')
table(['问题','首选方法','关键假设','备选方法'],[
 ('两独立疾病组的 thalach 是否不同','Welch t 检验，比较两组平均值。','人与人之间相互独立。每组样本大，均值近似正态问题不大，但要看看有没有严重离群值。不要求两组方差相等。','Mann–Whitney U 检验。注意它比的是“哪组的值更容易偏大”，不能直接说成中位数检验。'),
 ('同一患者前后测量是否变化','配对 t 检验：先算每个人“后−前”的差，再检验这些差值。','前后要正确配对，不同患者之间独立；样本小时，差值要大致正态、没有严重离群。','Wilcoxon 符号秩检验，要求差值分布大致对称；不对称时可以用符号检验。'),
 ('sex 与 disease 的比例是否有关','2×2 列联表 Pearson χ² 检验。','每人只落在一个格子里，观测独立，每格的期望频数够大。','格子频数太小时用 Fisher 精确检验。'),
 ('age 与 thalach 是否相关','Pearson 相关，报告 r 和置信区间。','关系大致是一条直线，观测独立，没有一两个极端点左右结果。','Spearman 秩相关，只要求单调，不要求直线。'),
 ('age、sex、thalach 能否预测 disease','Logistic 回归，以 disease 为结局，报告 OR 和 95% CI。','观测独立；连续变量和 log-odds 大致呈直线关系；没有完全分离和严重共线性；疾病例数相对变量数要够。','带惩罚的 Logistic 回归，可以加样条处理非线性。'),
], [3.4,4.0,5.3,4.7])
h('放到本题里要注意什么',2)
p('为什么直接选 Welch：很多人习惯先做方差齐性检验，再决定用哪种 t。但这种“两步走”会让整体错误率偏离 5%。Welch 在两组方差相等时和普通 t 几乎一样，没什么损失，所以我直接选它。',bold_label='为什么直接选 Welch：')
p('配对题：我本来想用数据实际算一下，结果发现 CSV 里每人只有一行，没有前后两次测量，所以这题只能选方法，没法算。',bold_label='配对题：')
p('相关题：我会先画散点图。相关系数接近 0 不代表没关系，可能是 U 形；而且相关也不等于因果。',bold_label='相关题：')
p('预测题：模型里系数显著，只说明有关联，不说明“预测得好”。要回答能不能预测，得把数据分成训练集和验证集，在模型没见过的数据上看 AUC、校准和 Brier 分数；换一个中心使用，还要在新中心重新验证。另外，这些变量和疾病标签是同时测的，只能说“区分当下有没有病”，不能说“预测将来会不会得病”。',bold_label='预测题：')
p('任务 3 只要求选方法，我没有把五题都算一遍。如果都算，就成了到处找显著结果，这正是任务 5 要讨论的风险。')

h('任务 4：报告一个主要结果')
h('先看样本和分布',2)
p(f"原始数据一共 {R['total_n']} 人，disease 没有缺失；但心率缺了 {R['excluded_n']} 人（{100*R['excluded_n']/R['total_n']:.2f}%），真正进入分析的是 {R['included_n']} 人。我只按分组和心率筛选，别的字段缺了不删。这里有个容易踩的坑：报告每组 n 时要用实际分析的 {G1['analysis_n']} 和 {G0['analysis_n']}，不能用原始的 {G1['total_n']} 和 {G0['total_n']}。")
table(['项目','无疾病证据组 disease=0','疾病组 disease=1'],[
 ('原始人数',G0['total_n'],G1['total_n']),
 ('心率缺失人数及比例',f"{G0['missing_n']}（{G0['missing_percent']:.2f}%）",f"{G1['missing_n']}（{G1['missing_percent']:.2f}%）"),
 ('实际分析人数 n',G0['analysis_n'],G1['analysis_n']),
 ('均值 ± 标准差，次/分',f"{G0['mean']:.2f} ± {G0['sd']:.2f}",f"{G1['mean']:.2f} ± {G1['sd']:.2f}"),
 ('中位数［第 1，第 3 四分位数］',f"{G0['median']:.1f}［{G0['q1']:.1f}，{G0['q3']:.1f}］",f"{G1['median']:.1f}［{G1['q1']:.1f}，{G1['q3']:.1f}］"),
 ('最小值至最大值，次/分',f"{G0['minimum']:.0f} 至 {G0['maximum']:.0f}",f"{G1['minimum']:.0f} 至 {G1['maximum']:.0f}"),
 ('偏度',f(G0['skew']),f(G1['skew'])),
], [5.4,6.0,6.0])
chol0 = zeros['chol_zero_by_site']
p(f"数据核查时还发现两件事。一是有 {len(dup)} 对记录（如 {dup[0][0]} 和 {dup[0][1]}，另 {len(dup)-1} 对在 VA Long Beach）所有临床字段一模一样，可能是重复录入，也可能只是巧合；我没法确认是同一个人，所以没删，但独立性需要找原始资料确认。二是有 {sum(chol0.values())} 条 chol=0（Switzerland {chol0['Switzerland']} 条全是 0，VA Long Beach {chol0['VA Long Beach']} 条）、{zeros['trestbps_zero']} 条 trestbps=0，生理上不可能，应该是没测、用 0 占了位。它们不影响这次的心率分析，但如果有人拿 chol 直接算平均值，就会算错。",size=9,before=4)
img('thalach-distribution.png','图 1　两组 thalach 分布（每格 10 次/分，两组纵轴刻度相同）',15.6)
p(f"分布检查：跑 t 检验之前我先画了图。t 检验只告诉我均值差，图还能看出分布形状、有没有奇怪的点、两组重叠多少。两组都是单峰，疾病组整体往左移，但重叠很多。无疾病证据组略左偏（偏度 {f(G0['skew'])}），疾病组基本对称（{f(G1['skew'])}）。按 1.5×IQR 规则，两组分别有 {len(out0)} 个和 {len(out1)} 个离群点（{fmt_list(out0)} 和 {fmt_list(out1)} 次/分），都在生理上说得通的范围内，按事先规定不删。每组都有 390 多人，均值差近似正态，t 检验可以放心用；不过记录之间是否独立，图上是看不出来的。",bold_label='分布检查：')
h('主要结果',2)
p(f"疾病组减无疾病证据组，均值差是 {f(P['mean_difference'])} 次/分，95% CI 为 {ci(P['ci_low'],P['ci_high'])} 次/分。标准误 {P['se']:.4f}，Welch t={f(P['t'],3)}，自由度约 {P['df']:.2f}，双侧 p={sci(P['p'])}。p 值小到这个程度，精确数字依赖分布尾部的近似，正式报告写 p<0.001 就够了；这里保留精确值，是为了方便核对。")
p(f"计算核对：为了确认软件给的方向和区间就是我想要的，我手算了一遍。标准误 = √({G1['sd']:.4f}²/{P['n1']} + {G0['sd']:.4f}²/{P['n0']}) = {P['se']:.4f}；t = {f(P['mean_difference'],4)}/{P['se']:.4f} = {f(P['t'],4)}；95% CI = 均值差 ± t 临界值 × 标准误，临界值按 Welch 自由度取。",size=9)
p(f"效应量：我先想到的是 Cohen’s d，算出来是 {f(P['cohen_d'],3)}（小样本修正后的 Hedges’ g 是 {f(P['hedges_g'],3)}），按惯例算“大效应”。但跟医生说“d 等于负 0.86”不好懂，说“平均低 20 次/分”马上就懂。所以主要效应量用原始单位的均值差，d 只作补充。",bold_label='效应量：')
p('H₀ 决策：p<0.05，拒绝“两组平均值相等”的 H₀。这里有两个容易说错的地方。一是 p 值不是“H₀ 为真的概率”，而是假设 H₀ 和检验条件都成立时，得到至少这么极端的 t 值的概率。二是 95% CI 不是“真实 Δ 有 95% 的概率落在这个区间里”，而是说按同样方法反复抽样、构造区间，大约 95% 的区间会盖住真实值。这个区间整段都小于 0，说明疾病组平均心率更低。',bold_label='H₀ 决策：')
h('统计关联、实际意义与因果边界',2)
p('统计关联：在测到心率的这些人里，两组的平均最大心率确实不同，而且证据很强。这是“平均值”层面的结论。',bold_label='统计关联：')
p(f"实际意义：平均差 {abs(P['mean_difference']):.1f} 次/分不算小，但资料没给“差多少才有临床意义”的标准，光凭显著不能说它有临床价值。图上两组重叠很多，我想把重叠量化：Mann–Whitney 的 U 除以两组人数之积，就是“随机各抽一人，疾病组那个人心率更高的概率”，算出来约 {100*MW['prob_disease_higher']:.0f}%（平局算一半）。也就是说，平均差了 20 次，仍有四分之一以上的配对是疾病组的人心率更高，所以不能拿一个人的心率去判断他有没有病。",bold_label='实际意义：')
p('因果边界：这是观察性的横断面资料，没有随机分组。我想到两个具体的替代解释：一是用药，β 受体阻滞剂这类药本身就会压低心率，而数据里没有用药信息；二是测量方式，thalach 是运动试验中达到的最大心率，如果疾病组的人因为胸痛等症状提前停了，测到的“最大”心率自然偏低。所以不能说“心脏病导致心率降低”，也不能说把心率提上去就能降低患病风险。',bold_label='因果边界：')

h('任务 5：风险与退出卡')
h('混杂因素与调整分析',2)
p('讲完主结果，我会问自己：这 20 次的差距，会不会是别的因素造成的？判断混杂因素我用一个标准：这个变量要同时和“分组”有关、又和“心率”有关。按这个标准，我挑了年龄、性别和来源中心。')
p(f"年龄：常用的经验是最大心率约等于 220 减年龄，年龄越大心率越低；同时年龄越大，心脏病也越常见。数据里疾病组平均 {B['1']['age_mean']:.2f} 岁，无疾病证据组 {B['0']['age_mean']:.2f} 岁，差了 5 岁多。调整模型里年龄系数是每岁 {f(cov['age']['coefficient'])} 次/分，和“220 减年龄”的经验很接近。",bold_label='年龄：')
p(f"性别：疾病组男性占 {B['1']['male_percent']:.2f}%，无疾病证据组占 {B['0']['male_percent']:.2f}%，构成差别很大。模型里性别系数不显著（p={cov['sex']['p']:.2f}），但我没有把它删掉：协变量是事先按专业知识定的，不能看了 p 值再决定放不放。",bold_label='性别：')
p('中心：这是我看数据时才意识到的。四个中心的疾病比例差得很远：'+'、'.join(f"{k} {v['disease_percent']:.1f}%" for k,v in SD.items())+f"，瑞士几乎全是疾病组；调整后，其他三个中心的平均心率比 Cleveland 低 {min(site_gaps):.0f}–{max(site_gaps):.0f} 次/分。中心既和分组有关、又和心率有关，不调整就会混进组间比较。",bold_label='中心：')
p(f"于是我拟合了 {A['formula']}。中心是名字不是数字，所以作为分类变量放进去；标准误用 HC3 稳健标准误，它不要求每个人的残差方差相同，和选 Welch 是同一个思路。{A['n']} 人这些变量都齐全。结果疾病系数为 {f(A['coefficient'])} 次/分，95% CI {ci(A['ci_low'],A['ci_high'])}，p={sci(A['p'])}。和未调整的 {f(P['mean_difference'])} 相比缩小了约 {100*attenuation:.0f}%，说明原来的差距里有相当一部分可以由年龄、性别和中心解释，但区间仍然不跨 0。调整后的数也不是因果效应，因为没测的因素（比如用药）模型调整不了。")
table(['来源中心','n₀ / n₁','均值差，次/分','95% CI'],[
 (s['site'],f"{s['n0']} / {s['n1']}",f(s['mean_difference']),ci(s['ci_low'],s['ci_high'])) for s in R['by_site']
], [4.3,2.6,3.9,6.6])
p(f"我还把四个中心分开算了一遍（没做多重比较校正，只作探索），四个点估计都是负的。看到 Switzerland 和 VA Long Beach 的区间跨 0，很容易以为“这两个中心没关联”，但这里有两个坑。一是不显著不等于没关联：瑞士无疾病证据组只有 8 人，区间宽得说明不了什么。二是“有的显著、有的不显著”不能证明中心之间不一样，要回答这个问题得做交互检验。我事后补做了 disease×site 交互的联合检验（HC3），F={IX['F']:.2f}，p={IX['p']:.3f}，证据不足以下结论。不过 VA Long Beach 的点估计确实明显偏小，所以合并的调整估计更像是各中心“平均”出来的结果。",before=4)
p(f"另外，Mann–Whitney U={MW['U']:,.0f}，双侧 p={sci(MW['p'])}，方向一致，但它比的是分布而不是均值。上面的模型假设各因素的作用可以相加、年龄是直线关系；HC3 也处理不了没测的混杂或中心内部的相关，所以敏感性分析不能证明偏倚已经排除。",size=9)
h('缺失与选择风险',2)
p('一开始我只是按中心例行数了一下心率缺失：'+'、'.join(f"{k} {v['missing_n']}/{v['total_n']}"+(f"（{v['missing_percent']:.1f}%）" if v['missing_percent']>10 else '') for k,v in MS.items())+f"。{R['excluded_n']} 条缺失里有 {MS['VA Long Beach']['missing_n']} 条在 VA Long Beach，太集中了，我就追问：这些人为什么会缺？回去看同一行的其他字段，发现这 {R['excluded_n']} 人的静息血压全都缺失，ST 压低也缺了 {MP['oldpeak_also_missing']} 个。也就是说，不是单单漏记了心率，而是整次检查的数据都没有，可能是没做或没做完检查。所以不能想当然地认为缺失是完全随机的。")
p(f"比较已观测特征：缺失者平均 {MP['missing']['age_mean']:.1f} 岁、男性占 {MP['missing']['male_percent']:.1f}%，纳入者是 {MP['observed']['age_mean']:.1f} 岁、{MP['observed']['male_percent']:.1f}%，差别主要是因为缺失几乎都来自 VA Long Beach。在 VA 中心内部比较更公平：缺失者的疾病比例是 {MP['va_missing']['disease_percent']:.1f}%，纳入者是 {MP['va_observed']['disease_percent']:.1f}%。缺失的人和纳入的人确实不一样，但看不出一个简单的方向。",bold_label='比较已观测特征：')
p(f"事后补充的敏感性分析：这会不会动摇主结果？我把 VA Long Beach 整个拿掉重算，均值差是 {f(EV['mean_difference'])} 次/分，95% CI {ci(EV['ci_low'],EV['ci_high'])}，和主结果几乎一样，说明主结果不是靠这个中心撑着的。但它回答不了“缺失的人如果有数据会怎样”。要真正解决，得先查清缺失原因，再在明确的假设下做多重插补和非随机缺失的情景分析，这次我没有做。",bold_label='事后补充的敏感性分析：')
p('说明：缺失比较、去掉 VA Long Beach 的分析和交互检验，是我看完第一版结果后补的，在分析约定里单独列成“事后补充”。它们没有改变主要终点、假设和结论，只用来检查结果稳不稳。',size=9)
h('只挑显著指标的风险',2)
p(f"假设团队测了 20 个指标，其实都和疾病无关，每个都用 α=0.05。每个不出错的概率是 0.95，20 个都不出错是 0.95²⁰≈0.36，所以至少冒出一个“显著”的概率约 {100*R['independent_20_tests_fwer']:.1f}%，平均会出 1 个假阳性。指标之间相关时这个数不精确，但风险还在。如果只挑那个显著的写进结论，读者看到的 p<0.05 就是误导，而且挑出来的效应往往还被高估了。")
p('我的应对是：一开始就把 thalach 定为唯一主要终点，其他分析都标成敏感性或探索性，并且如实报告。如果事先就要同时确认 20 个结论，可以用 Bonferroni（每个检验的阈值降到 0.05/20=0.0025）或更高效的 Holm 法控制家族错误率；如果是探索性筛选，可以控制 FDR，筛出来的发现要在新数据里再验证。')
h('退出卡四句话',2)
p('主张：在这批四中心资料里，有疾病证据的人，平均最大运动心率更低。',bold_label='主张：')
p(f"证据：疾病组 {P['n1']} 人、无疾病证据组 {P['n0']} 人，均值差 {f(P['mean_difference'])} 次/分，95% CI {ci(P['ci_low'],P['ci_high'])}，p={sci(P['p'])}；调整年龄、性别和中心后是 {f(A['coefficient'])} 次/分，95% CI {ci(A['ci_low'],A['ci_high'])}。",bold_label='证据：')
p('推理：主要检验的区间不包含 0，调整分析和四个中心的点估计方向也一致，所以数据支持两者之间存在统计关联。',bold_label='推理：')
p('边界：这是没有随机分组的横断面观察资料，中心差异、集中缺失、用药和运动试验提前终止这些因素都没排除，所以不能说是因果，也不能说有临床价值，更不能用来给个人下诊断。',bold_label='边界：')

doc.core_properties.title=title
doc.core_properties.subject='心脏病证据分组与最大运动心率的统计检验'
doc.core_properties.author=''
doc.core_properties.keywords='Welch t 检验, 假设设计, 学生作答'
doc.save(OUT/'统计检验与假设设计_学生完成版.docx')
(OUT/'学生作答.md').write_text('\n'.join(md),encoding='utf-8')
print(OUT/'统计检验与假设设计_学生完成版.docx')
