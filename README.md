# 2020_mirae_insurance_competition

---------------------

** Contributor
JongHwan Park : bomebug15@ds.seoultech.ac.kr
Dohyeon Lee : dohyeon2941@seoultech.ac.kr

- 미래에셋에서 주관하고 Programmers에서 주최하는 2020 금융 빅데이터 페스티벌에서 사용한 코드입니다.
(https://programmers.co.kr/competitions/252/2020-miraeasset)

저희가 참여한 주제는 보험금 청구 건 분류 문제였으며, 질병코드, 치료행위, 의료기관 등의 데이터를 분석하여 고객들의 상태에 따라 보험납부 여부를 즉시지급, 서류심사, 방문심사로 분류하는 문제였습니다. EDA 과정은 Eda_process.ipynb에 포함되어 있습니다. 저희는 Lightgbm 모델을 사용하였으며, pipeline을 통해 *main.py 에서 작동할 수 있도록 설계했습니다. 사용법은 다음과 같습니다

### How to use
--------------
```
python main.py --estimators 25000 --leaves 300 --lr 0.01 --topk 20
```
--------------
 또한 원본 데이터는 제외하고 업로드 하였습니다. 사용한 디렉토리는 다음과 같습니다.

* Folder structure    
=========    
-2020_mirae_insurance_competition    
|--modules    
|---Modeling    
|----main.py    
|----models.py    
|----preprocessing.py    
|--resources    
|---dataset    
|----insurance    
|--results    
|---submit    
|----final_submit.csv    
Eda_process.ipynb    
=========    

