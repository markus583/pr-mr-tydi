import argparse
import pandas as pd


# example usage: python merge_shards.py -f *_run.mrtydi.english.test.txt -o full_run.mrtydi.english.test.txt
def main():
    parser = argparse.ArgumentParser(
        description="Concatenate the results of multiple shards."
    )
    parser.add_argument(
        "-f", "--files", nargs="+", help="<Required> Files to search in", required=True
    )
    parser.add_argument(
        "-o", "--output", help="<Required> File to write to.", required=True
    )
    parser.add_argument(
        "-k",
        help="<Optional> How many results to return for each query [k = 100 by default].",
        required=False,
        default=100,
    )
    args = parser.parse_args()
    files = args.files
    output_file = args.output
    k = args.k

    shard_dfs = []
    print(files)
    for f in files:
        print(f)
        df = pd.read_csv(f, delimiter="\t", names=["qid", "docid", "score"])
        shard_dfs.append(df)

    full_df = pd.concat(shard_dfs, axis=0)
    sorted_df = (
        full_df.sort_values("score", ascending=False)
        .groupby("qid")
        .head(k)
        .sort_values("qid")
    )
    sorted_df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"Output has been written to {output_file}")


if __name__ == "__main__":
    main()
