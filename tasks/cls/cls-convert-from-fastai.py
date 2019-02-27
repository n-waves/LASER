from pathlib import Path

import fire
import pandas as pd

def convert_to_laser(fa_file, dest_file, val_dest_file=None):
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(fa_file, header=None)
    df = df.replace("\t", " ")
    df[1] = df[1].fillna(" ") +" "+ df[2].fillna(" ")
    df[1] = df[1].str.replace(r"[\t \n\r]+", " ", regex=True)
    del df[2]
    escapechar=None
    if val_dest_file is not None:
        val_sz = int(len(df) * 0.1)
        print("Extracting dev set 10%", val_sz)
        val_df = df.iloc[:val_sz]
        val_df.to_csv(val_dest_file, header=None, index=None, sep="\t", na_rep='', encoding="utf-8",
                      quoting=3, escapechar=escapechar)  # pd.csv.QUOTE_NONE
        df = df.iloc[val_sz:]
    df.to_csv(dest_file, header=None, index=None, sep="\t", na_rep='', encoding="utf-8", quoting=3, escapechar=escapechar)#pd.csv.QUOTE_NONE

langs = ['en', 'fr', "de", 'ja']
def main(fastai_dir, where="cls"):
    where = Path(where)
    fastai_dir = Path(fastai_dir)

    for lang in langs:
        print("Preparing", lang)
        size=1800
        convert_to_laser(fastai_dir / f"{lang}-books" / f"{lang}.train.csv", where / f"cls.train{size}.{lang}", where / f"cls.dev.{lang}")
        convert_to_laser(fastai_dir / f"{lang}-books" / f"{lang}.test.csv", where / f"cls.test.{lang}" )

if __name__ == "__main__":
    fire.Fire(main)