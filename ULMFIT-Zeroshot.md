# CLS
To generate pseudo lables  
```
source activate laser
export LASER=path/to/laser
export MULTIFIT=~/workspace/ulmfit-multilingual
cd ${LASER}
cd tasks/cls
python ./cls-convert-from-fastai.py ${MULTIFIT}/data/cls/
ls cls
# cls.dev.de  cls.dev.fr  cls.test.de  cls.test.fr  cls.train1800.de  cls.train1800.fr
# cls.dev.en  cls.dev.ja  cls.test.en  cls.test.ja  cls.train1800.en  cls.train1800.ja 

bash cls.sh
.....
# You should end up with models and a table showing you how well cross lingual classification worked  

# to generate laser, note .h5 extension is missleading it is actually a pickle format

for LANG in en de fr; do
    echo $LANG
    python ${LASER}/source/generate_pseudo_labels.py embed1800/cls.en-en.h5 ~/workspace/ulmfit-multilingual/data/cls/${LANG}-books --suffix=1 --dataset=cls| grep Test:
done
```


# MLDoc
To generate pseudo lables  
```
source activate laser
export LASER=path/to/laser
export MULTIFIT=~/workspace/ulmfit-multilingual
cd ${LASER}
cd tasks/mldoc
python ./mldoc-convert-from-fastai.py ${MULTIFIT}/data/mldoc/
ls MLDoc
# mldoc.dev.de  mldoc.test.de  mldoc.train10000.de  mldoc.train1000.de  mldoc.train2000.de  mldoc.train5000.de
# mldoc.dev.en  mldoc.test.en  mldoc.train10000.en  mldoc.train1000.en  mldoc.train2000.en  mldoc.train5000.en

bash mldoc.sh
.....
# You should end up with models and a table showing you how well cross lingual classification worked  

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
