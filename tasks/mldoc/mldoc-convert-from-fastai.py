from pathlib import Path

import fire
import pandas as pd

def convert_to_laser(fa_file, dest_file):
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(fa_file, header=None)
    df.to_csv(dest_file, header=None, index=None, sep="\t", na_rep='', encoding="utf-8", quoting=3)#pd.csv.QUOTE_NONE

langs = ['en', 'es', 'de', 'zh', 'fr', 'ru', 'ja', 'it']
def main(fastai_dir, where="MLDoc"):
    where = Path(where)
    fastai_dir = Path(fastai_dir)

    for lang in langs:
        print("Preparing", lang)
        convert_to_laser(fastai_dir / f"{lang}-10" / f"{lang}.train.csv", where / f"mldoc.train10000.{lang}")
        convert_to_laser(fastai_dir / f"{lang}-5" / f"{lang}.train.csv", where / f"mldoc.train5000.{lang}")
        convert_to_laser(fastai_dir / f"{lang}-2" / f"{lang}.train.csv", where / f"mldoc.train2000.{lang}")
        convert_to_laser(fastai_dir / f"{lang}-1" / f"{lang}.train.csv", where / f"mldoc.train1000.{lang}" )
        convert_to_laser(fastai_dir / f"{lang}-1" / f"{lang}.test.csv", where / f"mldoc.test.{lang}" )
        convert_to_laser(fastai_dir / f"{lang}-1" / f"{lang}.dev.csv", where / f"mldoc.dev.{lang}" )


if __name__ == "__main__":
    fire.Fire(main)