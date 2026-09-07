"""Entraînement du classifieur final sur les scores hybrides."""
from pathlib import Path
import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
ROOT=Path(__file__).resolve().parents[2]; INPUT=ROOT/"resultats/semantic_predictions.csv"; MODEL=ROOT/"modeles/similarity_classifier.joblib"; OUT=ROOT/"resultats/metrics/classification_report.txt"
def main():
 df=pd.read_csv(INPUT); X=df[["tfidf_score","semantic_score","hybrid_score"]]; y=df.label.astype(int)
 groups=df.doc_a.astype(str).str.extract(r"(\D+-?\d+)")[0].fillna(df.doc_a)
 tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(X,y,groups))
 m=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42); m.fit(X.iloc[tr],y.iloc[tr]); pred=m.predict(X.iloc[te]); prob=m.predict_proba(X.iloc[te])[:,1]
 text=classification_report(y.iloc[te],pred,digits=4)+f"\nConfusion matrix:\n{confusion_matrix(y.iloc[te],pred)}\nROC-AUC: {roc_auc_score(y.iloc[te],prob):.4f}\n"
 print(text); MODEL.parent.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,MODEL); OUT.write_text(text,encoding="utf-8")
if __name__=="__main__": main()
