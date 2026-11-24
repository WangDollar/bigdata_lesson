"""Teacher checks for the linear-model lesson. Outputs go to outputs/teacher-check/."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
BASE=Path(__file__).resolve().parents[2]  # lessons/08-linear-models
df=pd.read_csv(BASE/'data'/'heart.csv')
out=BASE/'outputs'/'teacher-check'; out.mkdir(parents=True,exist_ok=True)
summary={'rows':len(df),'positive':int(df.heart_disease_present.sum()),'negative':int((1-df.heart_disease_present).sum())}
pd.Series(summary).to_json(out/'data-summary.json',force_ascii=False)
print(summary)
