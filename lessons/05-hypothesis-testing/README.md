# 第 5 讲 统计检验与假设设计

> 状态：✅ 已导入

用四中心心脏病数据（920 人），把“心脏病证据和最大运动心率有关吗”改写成可检验的问题：定研究对象、比较对象和主要终点，写 H0/H1，选检验，报告效应量和置信区间，再讨论混杂、多重比较和因果边界。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/统计检验与假设设计.docx`](report/统计检验与假设设计.docx) | **交上去的答卷**（2026-10-14） |
| [`report/学生作答.md`](report/学生作答.md) | 答卷文本版，图 [`thalach-distribution.png`](report/thalach-distribution.png) |
| [`report/analysis-plan.md`](report/analysis-plan.md) | 计算前固定的分析约定 + 事后补充 |
| [`report/作答说明.md`](report/作答说明.md) | 作答文件说明 |

结论：疾病组平均最大心率低约 20.5 次/分；调整年龄、性别、中心后约 12.2 次/分。观察性资料，只说关联。

> `学生作答.md` 和 `build_answer.py` 是交卷之后又改过的版本：重新生成的 docx 比交上去的那份在任务 1 主要终点处多一句“看结果之前就定好了，而且只定这一个。”，其余文字完全相同。

## 目录

```
pbl/          课程材料：工作单（md + 两版 docx）、教师参考、教案、课件 pptx
  review/     教师数值核对（numeric_answers.md）和渲染脚本
data/         heart-disease-hypothesis.csv + README
src/homework/ analyze.py（分析）、build_answer.py（生成答卷 docx）
outputs/homework/  描述统计、缺失、分中心、调整模型 CSV + results.json
```

## 复现

在**本目录**下运行（需要 statsmodels 和 python-docx）：

```bash
pip install -r ../../requirements.txt
python -B src/homework/analyze.py        # → outputs/homework/、report/thalach-distribution.png
python -B src/homework/build_answer.py   # → report/统计检验与假设设计_学生完成版.docx、学生作答.md
```

## 原始来源

- `~/Desktop/note/l4/hypothesis` — 课程包，作答在 `student-solution/`
- `~/Desktop/note/l4/健康大数据学生工作表.docx`、`第4讲_统计检验与假设设计-Plan.pptx`

## 备注

- 讲次：文件夹叫 l4，课件叫“第4讲”，材料里写“第五章”，按实际上课顺序算作第 5 讲。
- 路径从 `student-solution/` 改到 `src/homework` + `outputs/homework` + `report`。用临时环境（statsmodels 0.15.0，和原来一致）重跑 `analyze.py`：6 个 CSV 和图逐字节一致，results.json 只有记录的数据绝对路径不同。
- 没导入 `review/reference-render/`（教师参考的渲染预览）。
