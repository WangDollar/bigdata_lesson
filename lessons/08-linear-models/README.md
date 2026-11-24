# 第 8 讲 线性模型、LDA 与 Logistic Regression

> 状态：✅ 课程材料已导入 · ⚠️ 没找到我自己的作业报告

以心脏病二分类数据（Statlog Heart，270 条，120 阳性 / 150 阴性，13 个混合类型变量）为案例，比较 logistic 与 LDA 的建模对象、概率来源、假设和参数量，再分析错误代价（5×FN+FP）、系数稳定性和因果边界。

## 作业报告

`report/` 目前是空的：`~/Desktop/note/8. Linear-models` 里只有课程包，桌面、Documents、Downloads 里也没找到第 8 讲的作业。如果作业在别处（比如交到了学习通、微信里），找到后放进 `report/`。

## 目录

```
pbl/          课程材料：工作单、教师参考、实验设计、WORKPLAN、slides/
  assets/     教学包数据和参考结果（packet A–D 的表和图）
  planning/   备课分析脚本和 2026-09-17 的 review
data/         heart.csv + README
src/
  linear-models-lab.ipynb                              简版实验 Notebook（工作单配套）
  pbl-linear-models-lda-logistic-regression-lab.ipynb  完整版 PBL Notebook（packet A–D）
  course/     教师检查脚本、生成课件脚本
```

## 复现

在**本目录**下运行：

```bash
pip install -r requirements.txt
cd src && jupyter lab                          # 两个 notebook 都从 src/ 打开
python -B src/course/teacher_analysis.py       # 教师检查，输出到 outputs/teacher-check/
python -B pbl/planning/build_review_notebook.py  # 重建 pbl/assets/ 和完整版 notebook
```

## 原始来源

`~/Desktop/note/8. Linear-models`

## 备注

- 路径调整：notebook 从 `notebooks/`、`notebook/` 合并到 `src/`，assets 移到 `pbl/assets/`，教师脚本输出从项目外 `.tmp/linear-models/` 改到 `outputs/`。两个 notebook、`teacher_analysis.py`、`build_review_notebook.py` 在新位置都跑通；重建出的 assets 与原文件只差换行符。
- 没导入 `7. Linear-models.zip`：是这个文件夹本身的打包备份。
