# CLS
To generate pseudo lables  
```bash
source activate laser
export LASER=path/to/laser
export MULTIFIT=~/workspace/ulmfit-multilingual
cd ${LASER}
cd tasks/cls
python ./cls-convert-from-fastai.py ${MULTIFIT}/data/cls/
#Preparing en
#Extracting dev set 10% 200
#Preparing fr
#Extracting dev set 10% 200
#Preparing de
#Extracting dev set 10% 200
#Preparing ja

ls cls
# cls.dev.de  cls.dev.fr  cls.test.de  cls.test.fr  cls.train1800.de  cls.train1800.fr
# cls.dev.en  cls.dev.ja  cls.test.en  cls.test.ja  cls.train1800.en  cls.train1800.ja 

bash cls.sh
# Training cls classifier (log files in embed1800)
#  - train on en, dev on en
#  - train on de, dev on de
#  - train on fr, dev on fr
# 
# Accuracy matrix:
# Train     en     de     fr
#  en:    84.55  84.15  83.90
#  de:    82.60  85.20  83.05
#  fr:    77.20  82.95  84.85
# You should end up with models and a table showing you how well cross lingual classification worked  

# to generate laser, note .h5 extension is missleading it is actually a pickle format

for LANG in en de fr; do
    echo $LANG
    python ${LASER}/source/generate_pseudo_labels.py embed1800/cls.en-en.h5 ~/workspace/ulmfit-multilingual/data/cls/${LANG}-books --suffix=1 --dataset=cls| grep Test:
done
#en
# | Test: 84.55% | classes: 47.85 52.15  0.00  0.00
#de
# | Test: 84.15% | classes: 48.35 51.65  0.00  0.00
#fr
# | Test: 83.90% | classes: 44.70 55.30  0.00  0.00

```


# MLDoc
To generate pseudo lables  
```bash
source activate laser
export LASER=path/to/laser
export MULTIFIT=~/workspace/ulmfit-multilingual
cd ${LASER}
cd tasks/mldoc

python ./mldoc-convert-from-fastai.py ${MULTIFIT}/data/mldoc/
#Preparing en
#Preparing es
#Preparing de

ls MLDoc
# mldoc.dev.de  mldoc.test.de  mldoc.train10000.de  mldoc.train1000.de  mldoc.train2000.de  mldoc.train5000.de
# mldoc.dev.en  mldoc.test.en  mldoc.train10000.en  mldoc.train1000.en  mldoc.train2000.en  mldoc.train5000.en

bash mldoc.sh

# Training MLDoc classifier (log files in embed1000)
#  - train on en, dev on en
#  - train on de, dev on de
#  - train on es, dev on es
#  - train on fr, dev on fr
#  - train on it, dev on it
#  - train on ru, dev on ru
#  - train on zh, dev on zh
# 
# Accuracy matrix:
# Train     en     de     es     fr     it     ru     zh
#  en:    91.48  87.65  75.48  84.00  71.18  66.58  76.65
#  de:    78.23  93.50  81.40  81.50  74.53  64.58  73.20
#  es:    71.62  84.00  93.73  78.90  73.38  53.33  55.83
#  fr:    81.30  88.75  80.12  90.85  72.58  67.35  79.40
#  it:    74.33  83.53  80.58  79.78  84.48  66.45  63.35
#  ru:    72.38  81.65  65.73  71.30  63.33  85.45  59.58
#  zh:    74.98  81.35  72.20  73.28  70.08  66.23  88.30

# to generate laser, note .h5 extension is missleading it is actually a pickle format
for LANG in en de es fr it ru zh; do                                        
    echo $LANG
    python ${LASER}/source/generate_pseudo_labels.py embed1000/mldoc.en-en.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-1 --suffix=1 --dataset=mldoc| grep Test:
done
#en
# | Test: 91.48% | classes: 23.77 24.90 26.25 25.07
#de
# | Test: 87.65% | classes: 21.98 24.45 27.65 25.93
#fr
# | Test: 84.00% | classes: 23.18 29.12 27.90 19.80
# ...
```
