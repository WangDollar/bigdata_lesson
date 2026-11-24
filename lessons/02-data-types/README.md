# 第 2 讲 变量类型与数据类型（理解数据）

> 状态：✅ 已导入

用 UCI 四中心心脏病历史数据（920 条：Cleveland 303、Hungary 294、Switzerland 123、VA Long Beach 200）为研究团队写一份数据使用建议书，内容包括：一行代表谁，各列能怎么用（测量尺度和变量角色），质量问题（缺失、编码、零值、中心差异）会影响什么，以及在什么条件下继续合并、什么条件下暂缓。WDBC 数据集作为高维和 PCA 的补充证据。

## 作业报告

| 文件 | 说明 |
|---|---|
| [`report/第2课理解数据.docx`](report/第2课理解数据.docx) | **作答完的个人探究工作单** |
| [`report/心脏数据科研思路讲稿.docx`](report/心脏数据科研思路讲稿.docx) | 翻转课堂 7–10 分钟口语稿（按五个问题展开） |
| [`report/稿子.docx`](report/稿子.docx) | 讲稿的另一版（开头接回整门课） |
| [`report/whiteboard-script.md`](report/whiteboard-script.md) | 白板讲稿：板书提纲、列名对照、扩展模块 |

结论：初步选 B（修订后合并使用），作为等待字典核查期间的暂定方案。`thal` 的 3/6/7 是编码，不能当数量用。最不确定的是四个中心的编码和采集流程是否一致；如果关键字段不可比，就改成 C（限制使用）。

## 目录

```
pbl/          工作单（v2 空白版）、证据包、设计说明、plan.docx、package-manifest.json
  assets/pbl-understanding-data/   packet B–D 的表和图（构建脚本生成）
  slides/     chap2_data.pptx（课件）
data/         heart-disease-four-site.csv（与 pbl/assets 里的是同一份）
src/
  pbl-understanding-data-lab.ipynb     实验 notebook（从 src/ 打开）
  course/pbl-understanding-data-build.py  教学包构建脚本
```

## 复现

在**本目录**下运行：

```bash
pip install -r ../../requirements.txt
cd src && jupyter lab                                  # 打开 notebook
python -B src/course/pbl-understanding-data-build.py   # 重建 pbl/ 下的 docx、assets、manifest 和 src/ 下的 notebook
```

注意：构建脚本会**覆盖** `pbl/` 下的 docx 和 assets。生成的工作单文件名是 `pbl-understanding-data-student-worksheet.docx`，课上实际发的是 v2 版（`v2_pbl-data-types-student-worksheet.docx`），v2 是在生成版的基础上手改的。

## 原始来源

`~/Desktop/note/l2`

## 备注

- 路径调整：`assets/` 移到 `pbl/assets/`，`notebook/` 并进 `src/`，构建脚本从 `scripts/` 移到 `src/course/`。脚本里的 `ROOT` 改成本讲的 `pbl/`，notebook 输出到 `src/`，manifest 里记录的路径也一起改了。
- 验证：notebook 在新位置跑通。构建脚本在副本里跑了一遍，内置校验通过（4 份 docx、34 个 assets、29 个 cell）。和原件相比：CSV、JSON 的数值全部一致，只有换行符不同；notebook 只差 assets 路径那一行；manifest 只差两条路径；6 张 PNG 内容一样，只有抗锯齿和一个像素的位置偏移，是 matplotlib 版本不同造成的。
