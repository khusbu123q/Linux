def unique_values(dict):
    unique_count=0
    result_key=None

    for key, values_list in dict.items():
        unique_elements=set(values_list)
        if len(unique_elements) > unique_count :
            unique_count=len(unique_elements)
            result_key=key
    return result_key

def main():
    dict={"khusbu":[2,6,5,4,3,1],"Gfg":[5,7,7,7,7,7,9,2,1,3,1,4,3,2],"is":[6,7,7,7],"best":[9,9,9,6,6]}
    print(unique_values(dict))

if __name__ == '__main__':
    main()