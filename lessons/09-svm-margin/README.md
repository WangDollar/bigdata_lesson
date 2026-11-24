# 第 9 讲 SVM 与 Margin

> 状态：✅ 已导入

在 UCI 脊柱生物力学数据（310 条，2 个预先指定的输入，训练/验证/测试 186/62/62）上，从八个点的几何开始，理解最大间隔、支持向量、soft margin 与 C、核函数与 gamma。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/第9讲作业_SVM与Margin.md`](report/第9讲作业_SVM与Margin.md) / `.docx` | 作业正文，图在 `report/figures/` |
| [`report/口语稿_白板讲解.md`](report/口语稿_白板讲解.md) / `.docx` | 讲台口语稿，约 8 分钟 |

版本历史（用 `git log -- lessons/09-svm-margin/report` 查看）：

| tag | 说明 |
|---|---|
| `l09-v2` | 第二版草稿 |
| `l09-v3-with-test` | 加入了锁定测试集评价 |
| `l09-final` | **最终提交**：测试集结果移出，等老师安排后再报告 |

## 目录

```
pbl/          课程材料：工作单、教师参考、实验设计、plan、四份 docx、slides/、student-release/
  assets/     教学包生成的数据审计、图和表
  review/     课程包的 QA 记录（渲染出来的 PDF/PNG 没放进来，可以用脚本重新生成）
data/         raw/ UCI 原档，student/ 训练+验证+测试特征，teacher/ 测试标签
src/
  svm-lab.ipynb   实验 Notebook（作业里的数字都来自它）
  make_figs.py    重画报告里的 7 张图
  verify2.py      补充核对：对偶系数、亚型、扰动
  build_docx.py   md → docx（需要 pandoc）
  course/         课程提供的脚本：lab_utils、数据整理、教师可行性检查、生成 notebook / 课件
```

## 复现

在**本目录**下运行（conda 环境 `svm-lab`，scikit-learn 1.8.0）：

```bash
pip install -r requirements.txt
jupyter lab src/svm-lab.ipynb                # 按工作单 A–G 运行
python -I src/make_figs.py                   # 重画 report/figures/
python -I src/verify2.py                     # 补充核对
python src/build_docx.py report/第9讲作业_SVM与Margin.md
```

Notebook 默认不读测试标签。

## 原始来源

- `~/Desktop/note/实践课作业/9. SVM` — 课程包和最终提交
- `~/Desktop/note/实践课作业/svm-作业工作文件` — 作业脚本和 v2 / v3 备份
- `~/Desktop/note/实践课作业/1.docx` — 早期口语稿草稿，没导入（最终版是 `report/口语稿_白板讲解.md`）

## 备注

- 路径从原来的 `scripts/`、绝对路径 `/Users/dollar/...`、项目外 `.tmp/svm/` 改成了本讲内的相对路径；`make_figs.py` 重画的 7 张图和原图逐字节一致，Notebook 和 `teacher_analysis.py` 在新位置跑通。
- 没导入：`planning/archive-v0-gpt-draft-20260918/`（30M，GPT 初稿的归档）、`review/rendered/`（31M，渲染出来的预览图）、`9. SVM.zip` 整包备份。原件都还在桌面上。
- `svm-pbl-intro-with-kernel.pptx`（加了核函数页的课件）放在 `pbl/slides/`。
