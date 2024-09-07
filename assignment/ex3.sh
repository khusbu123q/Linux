#!/bin/bash
BASE_URL="http://www.uniprot.org/uniprot/"
FASTA_LIST="fasta_files.txt"
MOTIF1="YVDRHPDDTINDYLNSI"
MOTIF2="MGNHTWDHPDIFEILTTK"
DOWNLOAD_DIR="fasta_files"
RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
if [ ! -f "$FASTA_LIST" ]; then
    echo "Error: File $FASTA_LIST does not exist."
    exit 1
fi
mkdir -p "$DOWNLOAD_DIR"
while IFS= read -r filename; do
    wget -q "${BASE_URL}${filename}" -P "$DOWNLOAD_DIR"
done < "$FASTA_LIST"
search_motifs() {
    local file="$1"
    local motif1="$2"
    loccontent=$(cat "$file")
    pos1=$(echo "$content" | grep -b -o "$motif1" | awk -F: '{print $1}')
    if [ ! -z "$pos1" ]; then
        echo "$file: $motif1 found at position $pos1" >> "$RESULT_DIR/results.txt"
    fi
    pos2=$(echo "$content" | grep -b -o "$motif2" | awk -F: '{print $1}')
    if [ ! -z "$pos2" ]; then
        echo "$file: $motif2 found at position $pos2" >> "$RESULT_DIR/results.txt"
    fi
}
for fasta_file in "$DOWNLOAD_DIR"/*.fasta; do
    search_motifs "$fasta_file" "$MOTIF1" "$MOTIF2"
done

echo "Search completed. Results are saved in $RESULT_DIR/results.txt."
