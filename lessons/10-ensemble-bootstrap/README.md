# 第 10 讲 集成学习与 Bootstrapping

> 状态：✅ 已导入 · 作业最终版为第二次做的 CTG 报告

多个弱模型怎样组合成更稳定的模型？在 CTG 胎心监护数据（2126 段 / 352 个录音）上，比较单树、bagging、随机森林、boosting 与 LR / SVM，用偏差-方差解释集成为什么有效。

## 作业报告

| 版本 | 文件 | 说明 |
|---|---|---|
| **最终版**（第二次做） | [`report/作业报告_集成学习与Bootstrapping_CTG.md`](report/作业报告_集成学习与Bootstrapping_CTG.md) / `.docx` | 四篇分析，脚本在 `src/`，图表在 `outputs/` |
| 第一次（修订版） | [`report/第10讲作业_集成学习与Bootstrapping.md`](report/第10讲作业_集成学习与Bootstrapping.md) / `.docx` | 图在 `report/figures/`，脚本在 `src/v1/` |
| 第一次（原始提交） | git tag `l10-v1` | `git checkout l10-v1 -- lessons/10-ensemble-bootstrap/report` |

## 目录

```
pbl/        课程材料：工作单、教师参考、workplan、课程材料说明、slides/
data/
  ctg/      方案 B（本作业用）：student/ 训练+验证+测试特征，teacher/ 测试标签，raw/ UCI 原始 zip
  wdbc/     方案 A：课件示例数据
src/        最终版分析脚本 part1–part4 + build_docx.py
  course/   课程提供的脚本：数据划分、课堂演示、重建课件
  v1/       第一次做的工作脚本和中间结果
outputs/    part1–part4 图表与结果表，report/ 为报告插图
report/     作业报告
```

## 复现

在**本目录**下运行（conda 环境 `svm-lab`，scikit-learn 1.8.0）：

```bash
pip install -r requirements.txt
python src/part1_single_tree.py
python src/part1_depth2_check.py
python src/part2_bagging.py
python src/part3_boosting.py
python src/part3_rounds_check.py
python src/part4_fair_comparison.py
python src/part4_protocol.py
python src/part4_importance.py
python src/part4_followup.py
python src/build_docx.py          # 重建报告 docx，需要 pandoc；装了 Word 会顺便刷新目录
```

所有分析只读 `data/ctg/student/`，随机种子 1010，锁定测试集没有打开。

## 原始来源

- `~/Desktop/作业/10. Ensemble` — 第二次做的（最终版）
- `~/Desktop/note/实践课作业/10. Ensemble/作业提交*`、`实践课作业/ensemble-作业工作文件` — 第一次做的

## 备注

- 导入时把原来的 `analysis/`、`data-ctg/`、`scripts/` 路径改成了现在的结构，脚本重跑结果与原始输出逐字节一致。
- 课程脚本 `demo_ensemble.py` 原来用 `parents[4]` 会写到项目外（报告第 9 节第 13 条记录过），已改为写到 `outputs/course-demo/`。
