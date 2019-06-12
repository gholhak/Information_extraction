from data_utils import DataHandler

dh_obj = DataHandler()


def main():
    _data_to_load = 'data\\test_fold1.txt'
    filename_to_save = "test_fold.csv"
    list_data = dh_obj.load_txt_data_as_list(_data_to_load)
    ner_data = dh_obj.extract_tagged_tokens_as_csv(list_data, filename_to_save)
    print(ner_data)


if __name__ == '__main__':
    main()
