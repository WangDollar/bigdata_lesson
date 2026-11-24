# 数据说明

来源：GEO [GSE49711](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49711)（SEQC 神经母细胞瘤 RNA-seq），表达值为 log2(1+FPKM)，按基因汇总。

| 文件 | 内容 | 入库 |
|---|---|---|
| `cohort120.npz` | 固定的 120 人 × 23,146 基因表达矩阵（17M） | ❌ 超过 10MB |
| `patient-manifest.csv` | 患者名单、临床注释和 80/40 划分 | ✅ |
| `gene-dictionary.csv` | 基因字典 | ✅ |
| `source-metadata.csv` | 样本来源元数据 | ✅ |
| `source/` | GEO 的 README、series/samples SOFT 注释和 source-ledger | ✅ |

**获取 `cohort120.npz`**：从课程包 `~/Desktop/note/l3/student-package/data/cohort120.npz` 复制到本目录。原始表达文件 `GSE49711_SEQC_NB_TUC_G_log2.txt.gz`（约 78MB）课堂实验用不到，要自己核对原始数据的话从 GEO 下载。

高危标签是研究中的临床风险分组，不是“健康 / 患病”的诊断分类。
