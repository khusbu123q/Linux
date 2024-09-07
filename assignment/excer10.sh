#!/bin/bash

DNA_SEQUENCE="ACGTACGTGACCGTAAGCT"

count_nucleotides() {
    local sequence="$1"
    local valid_nucleotides="ACGT"
    
    A_count=0
    C_count=0
    G_count=0
    T_count=0
    
    if [[ ! "$sequence" =~ ^[ACGT]*$ ]]; then
        echo "Error"
        exit 1
    fi
    
    for nucleotide in $(echo "$sequence" | fold -w1); do
        case "$nucleotide" in
            A) A_count=$((A_count + 1)) ;;
            C) C_count=$((C_count + 1)) ;;
            G) G_count=$((G_count + 1)) ;;
            T) T_count=$((T_count + 1)) ;;
        esac
    done
    
    # Print results
    echo "A: $A_count"
    echo "C: $C_count"
    echo "G: $G_count"
    echo "T: $T_count"
}
count_nucleotides "$DNA_SEQUENCE"
