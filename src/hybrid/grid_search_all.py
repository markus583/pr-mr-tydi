# list: {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
import subprocess

languages = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]
for lang in languages:
    print(lang)
    subprocess.call(
        f"python grid_search.py --language {lang} "
        f"--dense_runfile ../../runs/dense/run.mdpr.mrtydi-v1.1-{lang}.dev.txt"
        f" --bm25_runfile ../../runs/sparse/run.bm25.mrtydi-v1.1-{lang}.dev.txt "
        f"--output_directory ../../runs/hybrid/{lang}",
        shell=True,
    )
print("Done")
