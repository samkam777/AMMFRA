import pandas as pd
import numpy as np


data_path = "/home/jadeting/samkam/code/dataset/data/"



if __name__ == '__main__':
    train_data = pd.read_csv(data_path + "/new_dataset4/train_data.csv", delimiter=",")
    print(train_data)
    train_data['target_item_id'] = train_data.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    print(train_data)
    train_data['target_item_rating'] = train_data.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    print(train_data)
    train_data.to_csv(data_path + "/new_dataset4/train_data_other_methods.csv", index=False)

    test_data = pd.read_csv(data_path + "/new_dataset4/test_data.csv", delimiter=",")
    print(test_data)
    test_data['target_item_id'] = test_data.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    print(test_data)
    test_data['target_item_rating'] = test_data.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    print(test_data)
    test_data.to_csv(data_path + "/new_dataset4/test_data_other_methods.csv", index=False)

    val_data = pd.read_csv(data_path + "/new_dataset4/val_data.csv", delimiter=",")
    print(val_data)
    val_data['target_item_id'] = val_data.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    print(val_data)
    val_data['target_item_rating'] = val_data.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    print(val_data)
    val_data.to_csv(data_path + "/new_dataset4/val_data_other_methods.csv", index=False)





