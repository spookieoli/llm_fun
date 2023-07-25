from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
from typing import Tuple


class DataSetCreator:
    """
    This class is used to create the data set for the model.
    """

    def __init__(self, path: str = "", instruction: str = "", result_str: str = "",
                 tokenizer: AutoTokenizer = None, model: object = None) -> None:
        """
        This method initializes the DataSetCreator class.
        :param path: The path to a JSON file that contains the data (Instruction + Result)
        :param instruction: The Instruction as a string
        :param result_str: The Result as a string
        :param tokenizer: The tokenizer that is used to tokenize the data
        NOTE: Your *.JSON file should have the following structure:
        field1: name = type -> values = (train, validation, test)
        field2: name = inputs_ids -> values = instruction + additional text
        field3: name = labels -> values = result
        """
        self.instruction = instruction
        self.result_str = result_str
        self.model = model
        self.tokenizer = tokenizer
        self.check_tokenizer()
        self.path = path
        self.__df = self.__read_data()
        self.__ds_orig, self.__ds = self.__create_datasets()

    def __read_data(self) -> pd.DataFrame:
        """
        Reads the data from the JSON file and returns it as a DataFrame.
        """
        return pd.read_json(self.path)

    def __create_datasets(self) -> Tuple[DatasetDict, DatasetDict]:
        """
        This method is used to create the data sets.
        """
        self.train_dataset = Dataset.from_pandas(self.__df[self.__df['type'] == 'train'])
        self.valid_dataset = Dataset.from_pandas(self.__df[self.__df['type'] == 'validation'])
        self.test_dataset = Dataset.from_pandas(self.__df[self.__df['type'] == 'test'])
        return DatasetDict(
            {'train': self.train_dataset, 'validation': self.valid_dataset, 'test': self.test_dataset}).remove_columns(
            ['type', '__index_level_0__']), DatasetDict(
            {'train': self.train_dataset, 'validation': self.valid_dataset, 'test': self.test_dataset}).remove_columns(
            ['type', '__index_level_0__']).map(self.tokenize, batched=True)

    def tokenize(self, example) -> str:
        """
        This method is used to tokenize the data.
        :param example: The example that should be tokenized
        :return: str
        """
        # iterate over all fields in the example
        for field_name in example:
            example[field_name] = self.tokenizer(example[field_name], padding="max_length", truncation=True,
                                                 return_tensors="pt").input_ids
        return example

    def get_ds(self) -> Tuple[DatasetDict, DatasetDict]:
        """
        This method returns the data set.
        :return: Dataset
        """
        return self.__ds_orig, self.__ds

    def check_tokenizer(self) -> None:
        """
        This method is used to check if the tokenizer has a pad_token.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
