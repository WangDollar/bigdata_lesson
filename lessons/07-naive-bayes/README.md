# 第 7 讲 机器学习基础与朴素贝叶斯

> 状态：✅ 已导入

用症状数据学预测模型，并判断它预测得好不好。数据是 UCI Acute Inflammations（120 例，6 个二值症状），用 BernoulliNB 预测膀胱炎症；七项核心任务：定义模型、训练与预测、三类数据、手算 Bayes、损失、分类评价、accuracy 的局限。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/第7讲作业_朴素贝叶斯.docx`](report/第7讲作业_朴素贝叶斯.docx) | 作业正文，由 `src/homework/build_homework.py` 生成 |
| [`report/第7讲讲稿.docx`](report/第7讲讲稿.docx) | 板书设计 + 口语讲稿 + 备答，由 `src/homework/build_script.py` 生成 |
| `report/figures/` | 3 张图（log loss、阈值、重复样本），png + pdf |

作业流程：`run_dev.py` 跑开发集数字 → `freeze.py` 冻结方案（`frozen-protocol.json`）→ notebook 打开锁定测试 → `make_figures.py` 出图 → `build_homework.py` / `build_script.py` 生成 docx。

## 目录

```
pbl/          课程材料：工作单、教师参考、实验设计、WORKPLAN、数据集推荐、四份 docx、slides/
  assets/     教学包数据、参考结果和公式图
  planning/   备课归档、review、原始数据
  review/     课程包 QA 记录（渲染预览图没放进来）
data/         raw/ UCI 原档，student/ 训练+验证+测试特征，teacher/ 测试标签
src/
  naive-bayes-lab.ipynb   学生 Notebook（干净版，测试默认关闭）
  homework/               我的作业脚本 + 已运行的 notebook（打开了锁定测试）
  course/                 课程脚本：lab_utils、数据整理、教师分析、教学包构建、完整版 PBL notebook
```

## 复现

在**本目录**下运行（conda 环境 `naive-bayes`，需要 python-docx）：

```bash
pip install -r requirements.txt
python -B src/homework/run_dev.py          # 开发集上的全部数字
python -B src/homework/make_figures.py     # 重画 report/figures/（需要 ~/.claude/skills/nature-figure）
cd src/homework && python -B build_homework.py && python -B build_script.py
```

## 原始来源

- `~/Desktop/note/l6/6. Naive-Bayes` — 课程包
- `~/Desktop/note/l6/6. Naive-Bayes/我的作业` — 作业

## 备注

- 讲次：文件夹叫 l6、上课时是第 7 讲（作业也写的第 7 讲），按实际的来。
- 路径从 `scripts/`、`notebook(s)/`、`我的作业/`、项目外 `.tmp/naive-bayes/` 改到本讲内。验证结果：
  - `make_figures.py` 重画的 png 与原图逐字节一致，pdf 只差文件里记录的生成时间；
  - `build_homework.py`、`build_script.py` 重建的 docx 文字与原件完全一致；
  - 三个 notebook、`teacher_analysis.py` 在新位置跑通；
  - 教学包构建脚本重建的 assets 只差换行符、库版本号和浮点末位。
- 没导入：`我的作业/work/render/`（docx 渲染出来的预览图和中间版本 PDF，可以重新生成）、`review/render-*`、`review/runtime-work`（21M 预览图）、`Naive Bayes.zip`（整包备份）。
