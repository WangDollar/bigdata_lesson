# 第 6 讲 多重检验校正

> 状态：✅ 已导入

做了很多检验，哪些发现值得验证？在 Golub 白血病表达数据（72 例，7,129 个探针，AML vs ALL）上比较原始 p、Bonferroni 和 BH，配合已知真值模拟区分发现数量、假发现与检出力。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/第五课作业_多重检验校正.md`](report/第五课作业_多重检验校正.md) / `.html` / `.pdf` | 作业正文（标题写的“第五课”，是当时的叫法），图在 `report/figures/` |
| [`report/翻转课堂讲稿与追问卡.md`](report/翻转课堂讲稿与追问卡.md) | 翻转课堂讲稿 |
| `report/slides/` | 投屏版（html/pdf）、A4 讲稿、讲解提纲.docx，图在 `slides/figs/` |

报告里所有数字都来自 `outputs/homework/results.json`。

## 目录

```
pbl/        课程材料：工作单、教师参考、实验设计、multiple-test.md、PBL评分标准表、slides/
data/       leukemia.csv + OpenML 元数据
src/
  multiple-test-lab.ipynb   学生实验 Notebook
  homework/
    analysis.py             作业全部分析（SHA-256 校验数据，固定协议）→ outputs/homework/、report/figures/
    export_pdf.py           md → html/pdf（用 Chrome 打印）
    slides/                 生成投屏版和讲解提纲
outputs/homework/           分析结果 CSV + results.json
```

## 复现

在**本目录**下运行：

```bash
pip install -r requirements.txt
python -B src/homework/analysis.py           # 约 6 秒
python -B src/homework/export_pdf.py         # 需要 Google Chrome
python -B src/homework/slides/build_slides.py
python -B src/homework/slides/build_word.py  # 需要 python-docx
```

## 原始来源

- `~/Desktop/note/l5/6. multiple-test` — 课程包
- `~/Desktop/note/l5/homework` — 作业
- `~/Desktop/note/l5/PBL评分标准表.docx`

## 备注

- 讲次：文件夹叫 l5，课程包和上课都是第 6 讲，作业标题里写的“第五课”，按实际上课顺序算作第 6 讲。
- 路径从 `../6. multiple-test/`、`homework/output/`、项目外 `.tmp/multiple-test/` 改到本讲内。`analysis.py` 重跑：11 个 CSV 逐字节一致，6 张图逐像素一致，results.json 只有时间戳、耗时和库版本不同。Notebook 在新位置跑通。
- `作业.pdf` 改名为 `第五课作业_多重检验校正.pdf`，和 `export_pdf.py` 的输出名一致。
- 没导入 `5-multiple-test.zip`（整包备份）。
