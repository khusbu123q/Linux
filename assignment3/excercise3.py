def similarity(seq1, seq2):
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)
def main():
    org1 = ["ACGTTTCA", "AGGCCTTA", "AAAACCTG"]
    org2 = ["AGCTTTGA", "GCCGGAAT", "GCTACTGA"]

    threshold = 0.4


    similar_pairs = [(seq1, seq2) for seq1 in org1 for seq2 in org2 if similarity(seq1, seq2) > threshold]
    if similar_pairs:
        print("Similar pairs:")
        for pair in similar_pairs:
            print(pair)
    else:
        print("No similar pairs found above the threshold.")

if __name__ == '__main__':
    main()