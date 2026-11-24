# 第 4 讲 高维数据与降维（PCA / UMAP）

> 状态：✅ 已导入

用 GEO GSE49711 中同一批 120 名神经母细胞瘤患者、23,146 个基因的表达数据（高危 41 / 非高危 79），讨论 p ≫ n 时距离和近邻还靠不靠得住、训练高分和未见患者表现差多少，以及 PCA 和 UMAP 哪个适合当主要展示。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/高维数据与降维_学生工作页作答.docx`](report/高维数据与降维_学生工作页作答.docx) | **作答完的学生工作页**，原文件名 `1.docx` |
| [`report/课堂讲稿_Q1-Q6与四道作业.docx`](report/课堂讲稿_Q1-Q6与四道作业.docx) | 翻转课堂口头讲稿，Q1–Q6 加四道短作业 |

结论：初始暂选 B，完成 Q1–Q6 后仍选 B，用 PCA 图作为主要展示，并说明压缩损失（二维 22.76%）；UMAP 只作辅助探索。如果任务重点变成近邻检索，或者 PCA 遗漏了关键的患者关系，就重新评估 C，或者暂选 D。

## 目录

```
pbl/          学生工作页（空白）、证据包、plan.docx、课程材料说明.md（原 README）
  readings/   背景阅读 PDF（R1–R6，来源见 readings/README.md）
data/         patient-manifest.csv、gene-dictionary.csv、source-metadata.csv、source/（GEO SOFT 注释）
              cohort120.npz（17M，不入库，见 data/README.md）
src/          课堂实验 notebook
requirements.txt  本讲固定版本依赖（含 umap-learn）
```

## 复现

先把 `cohort120.npz` 放进 `data/`，然后在**本目录**下运行：

```bash
pip install -r requirements.txt
cd src && jupyter lab   # 打开 pbl-high-dimension-reduction-classroom.ipynb
```

图会输出到 `outputs/classroom/`。第一次运行 UMAP 需要编译，大约多花一两分钟。

## 原始来源

- `~/Desktop/note/l3/student-package`：课程包、作答 `1.docx` 和讲稿
- `~/Desktop/note/l3/plan.docx`

## 备注

- 讲次：文件夹叫 l3，材料里写“第四课”，算作第 4 讲。
- notebook 改了两处路径：输出目录从 `assets/pbl-high-dimension-reduction/` 改成 `outputs/classroom/`，numba 缓存从 `review/numba-cache` 改成 `outputs/.numba-cache`（不入库）。读数据的写法本来就是“当前目录或上一级找 `data/`”，没动。
- 验证：用 requirements.txt 里固定的版本建了一个临时环境，在新位置把 notebook 跑了一遍。去掉字体警告和 stderr 以后，各个 cell 的数值输出和原 notebook 里保存的完全一致。
- 没导入：`.opencode-router/`（空目录）。
