import fire
import urllib.request
from pathlib import Path

mldoc=Path("MLDoc")
mldoc.mkdir(parents=True, exist_ok=True)
code_map={
  'spanish':'es',
  'chinese':'zh',
  'french':'fr',
  'japanese':'ja',
  'german':'de',
  'italian':'it',
  'russian':'ru',
  'english':'en'
}
for lang in ['spanish', 'chinese', 'french', 'japanese', 'german', 'italian', 'russian', 'english']:
  for ext in ['train.1000', 'train.2000', 'train.5000', 'train.10000', 'dev', 'test']:
    url = f'https://storage.googleapis.com/ulmfit/rcv/raw/{lang}.{ext}'
    print(url)
    urllib.request.urlretrieve(url, mldoc/f"mldoc.{ext.replace('.','')}.{code_map[lang]}")