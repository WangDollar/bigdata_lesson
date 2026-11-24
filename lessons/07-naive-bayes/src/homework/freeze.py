import json, datetime, pandas as pd
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]  # lessons/07-naive-bayes
train = pd.read_csv(BASE / 'data/student/train.csv')
freeze = dict(target='bladder_inflammation',
              features=['fever_ge_38','nausea','lumbar_pain','urine_pushing','micturition_pain','burning_urethra'],
              model='BernoulliNB(alpha=1, binarize=None)', alpha=1, fever_cutoff=38,
              threshold=0.5, threshold_rule='validation: sensitivity>=0.90, then max specificity, then closest to 0.5, then smaller',
              training_file='data/student/train.csv', training_ids=train.sample_id.tolist(),
              refit_train_plus_validation=False,
              frozen_at=datetime.datetime.now().isoformat(timespec='seconds'))
(Path(__file__).parent / 'frozen-protocol.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding='utf-8')
print(freeze['frozen_at'], freeze['threshold'], len(freeze['training_ids']))
