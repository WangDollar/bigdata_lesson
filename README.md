# 健康大数据与智能医学 · 实践课

这里放这门课每讲的 PBL 材料、分析代码和作业报告，每讲一个目录。

## 课程进度

| 讲 | 主题 | 目录 | 状态 |
|---|---|---|---|
| 2 | 变量类型与数据类型 | [`lessons/02-data-types`](lessons/02-data-types) | 🚧 待导入 |
| 4 | 高维数据与降维（PCA / UMAP） | [`lessons/04-high-dimension-reduction`](lessons/04-high-dimension-reduction) | 🚧 待导入 |
| 5 | 统计检验与假设设计 | [`lessons/05-hypothesis-testing`](lessons/05-hypothesis-testing) | 🚧 待导入 |
| 6 | 多重检验校正 | [`lessons/06-multiple-testing`](lessons/06-multiple-testing) | 🚧 待导入 |
| 7 | 机器学习基础与朴素贝叶斯 | [`lessons/07-naive-bayes`](lessons/07-naive-bayes) | 🚧 待导入 |
| 8 | 线性模型、LDA 与 Logistic Regression | [`lessons/08-linear-models`](lessons/08-linear-models) | 🚧 待导入 |
| 9 | SVM 与 Margin | [`lessons/09-svm-margin`](lessons/09-svm-margin) | 🚧 待导入 |
| 10 | 集成学习与 Bootstrapping | [`lessons/10-ensemble-bootstrap`](lessons/10-ensemble-bootstrap) | 🚧 待导入 |

> 讲次编号按实际上课用的（和材料 README 一致），比 plan.docx 里的大纲编号大 1。

## 每讲目录结构

```
lessons/NN-主题/
├── README.md        # 本讲说明、数据来源、怎么运行
├── pbl/             # 课堂材料：工作单、教师参考、slides、实验设计
├── data/            # 数据 + data/README.md（大文件不入库）
├── src/             # 分析脚本、notebook
├── outputs/
│   ├── figures/     # 生成的图
│   └── results/     # 生成的表 / json
└── report/          # 作业报告（md / docx / pdf）
```

## 约定

- 一讲一个目录，命名 `NN-英文短名`，两位数编号保证排序正常。
- 只提交原件和代码；`.zip` 打包件、`__pycache__`、日志、Office 临时文件不入库。
- 超过 10MB 的数据不入库，写在 `data/README.md` 里说明来源和获取方式。
- 报告的不同版本用 git 管理，不要再存 `-v2`、`_修订版` 副本。
- commit 信息格式：`[L09] 导入 SVM 作业报告`。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
