# FakeShield — Fake News Detector
# ================================

## EXACT STEPS TO RUN (copy-paste these)

### 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        ← you should see (venv) in terminal

### 2. Install dependencies
pip install -r requirements.txt

### 3. Prepare your dataset
# Place WELFake_Dataset.csv in the data/ folder, then:
python -c "import pandas as pd; df=pd.read_csv('data/WELFake_Dataset.csv'); df['text']=df['title'].fillna('')+' '+df['text'].fillna(''); df['label']=df['label'].astype(int); df=df[['text','label']].dropna(); df=pd.concat([df[df['label']==0].sample(2000,random_state=42),df[df['label']==1].sample(2000,random_state=42)]).sample(frac=1,random_state=42).reset_index(drop=True); df.to_csv('data/train.csv',index=False); print('Done:',len(df))"

### 4. Train the model (~20 min on CPU)
python train.py --epochs 2 --batch 8

### 5. Run the web app
python app.py
# Open: http://localhost:5000

### 6. Evaluate (optional)
python src/evaluate.py --all

## FILES EXPLAINED
config.py          ← all settings (change MAX_SEQ_LENGTH, EPOCHS, etc.)
train.py           ← trains the DistilBERT model
app.py             ← Flask web application
src/dataset.py     ← data loading and preprocessing
src/model.py       ← DistilBERT + classification head
src/predict.py     ← inference engine + CLI
src/evaluate.py    ← confusion matrix, ROC curve, metrics
templates/index.html ← web UI

## IMPORTANT NOTES
- Always activate venv before running anything
- Run commands ONE AT A TIME, not combined with >>
- If stuck: venv\Scripts\python.exe script.py  (use explicit python path)
