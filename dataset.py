import os
import tempfile
from PIL import Image  # Add the PIL library to handle images
import pandas as pd
from sedna.datasources import CSVDataParse, TxtDataParse, JSONDataParse

from core.common import utils
from core.common.constant import DatasetFormat

class Dataset:
    """
    Data:
    provide the configuration and handle functions of dataset.

    Parameters
    ----------
    config : dict
         config of dataset, include: train url, test url and label, etc.
    """

    def __init__(self, config):
        self.train_url: str = ""
        self.test_url: str = ""
        self.label: str = ""
        self._parse_config(config)

    def _check_fields(self):
        self._check_dataset_url(self.train_url)
        self._check_dataset_url(self.test_url)

    def _parse_config(self, config):
        for attr, value in config.items():
            if attr in self.__dict__:
                self.__dict__[attr] = value

        self._check_fields()

    @classmethod
    def _check_dataset_url(cls, url):
        if not utils.is_local_file(url) and not os.path.isabs(url):
            raise ValueError(f"dataset file({url}) is not a local file and not a absolute path.")

        file_format = utils.get_file_format(url)
        
        # Add support for image formats like .jpg and .png
        supported_formats = [v.value for v in DatasetFormat.__members__.values()] + ['jpg', 'jpeg', 'png']
        if file_format not in supported_formats:
            raise ValueError(f"dataset file({url})'s format({file_format}) is not supported.")

    @classmethod
    def _process_txt_index_file(cls, file_url):
        """
        convert the index info of data from relative path to absolute path in txt index file
        """
        flag = False
        new_file = file_url
        with open(file_url, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                if not os.path.isabs(line.split(" ")[0]):
                    flag = True
                    break
        if flag:
            root = os.path.dirname(file_url)
            tmp_file = os.path.join(tempfile.mkdtemp(), "index.txt")
            with open(tmp_file, "w", encoding="utf-8") as file:
                for line in lines:
                    line = line.strip()
                    words = line.split(" ")
                    length = len(words)
                    words[-1] = words[-1] + '\n'
                    for i in range(length):
                        file.writelines(
                            f"{os.path.abspath(os.path.join(root, words[i]))}")
                        if i < length-1:
                            file.writelines(" ")

            new_file = tmp_file

        return new_file

    def _process_index_file(self, file_url):
        file_format = utils.get_file_format(file_url)
        if file_format == DatasetFormat.TXT.value:
            return self._process_txt_index_file(file_url)
        if file_format == DatasetFormat.JSON.value:
            return file_url

        return None

    def process_dataset(self):
        """
        process dataset:
        process train dataset and test dataset for testcase;
        e.g.: convert the index info of data from relative path to absolute path
              in the index file(e.g.: txt index file).
        """

        self.train_url = self._process_index_file(self.train_url)
        self.test_url = self._process_index_file(self.test_url)

    @classmethod
    def _read_image_file(cls, image_file):
        """Read an image file and return the image object"""
        try:
            with Image.open(image_file) as img:
                return img.convert('RGB')  # Convert to RGB format
        except Exception as e:
            raise ValueError(f"Error reading image file {image_file}: {str(e)}")

    @classmethod
    def _write_image_file(cls, image, output_file):
        """Write the processed image to an output file"""
        image.save(output_file, format='JPEG')  # Save as JPEG

    @classmethod
    def _read_data_file(cls, data_file, data_format):
        data = None

        if data_format == DatasetFormat.TXT.value:
            with open(data_file, "r", encoding="utf-8") as file:
                data = [line.strip() for line in file.readlines()]

        if data_format == DatasetFormat.CSV.value:
            data = pd.read_csv(data_file)

        # Add support for image files
        if data_format in ['jpg', 'jpeg', 'png']:
            data = cls._read_image_file(data_file)

        return data

    def _get_dataset_file(self, data, output_dir, dataset_type, index, dataset_format):
        data_file = self._get_file_url(output_dir, dataset_type, index, dataset_format)

        if dataset_format in ['jpg', 'jpeg', 'png']:
            self._write_image_file(data, data_file)  # Handle image saving
        else:
            self._write_data_file(data, data_file, dataset_format)

        return data_file

    def split_dataset(self, dataset_url, dataset_format, ratio, method="default",
                      dataset_types=None, output_dir=None, times=1):
        """
        split dataset:
        Handles splitting both text and image datasets.

        Returns
        -------
        list
            the result of splitting dataset.
            e.g.: [("/dataset/train.txt", "/dataset/eval.txt")]
        """

        if method == "default":
            return self._splitting_more_times(dataset_url, dataset_format, ratio,
                                              data_types=dataset_types,
                                              output_dir=output_dir,
                                              times=times)

        raise ValueError(f"dataset splitting method({method}) is not supported,"
                         f"currently, method supports 'default'.")

    @classmethod
    def _splitting_more_times(cls, data_file, data_format, ratio,
                              data_types=None, output_dir=None, times=1):
        if not data_types:
            data_types = ("train", "eval")

        if not output_dir:
            output_dir = tempfile.mkdtemp()

        all_data = cls._read_data_file(data_file, data_format)

        data_files = []

        all_num = len(all_data)
        step = int(all_num / times)
        index = 1
        while index <= times:
            if index == times:
                new_dataset = all_data[step * (index - 1):]
            else:
                new_dataset = all_data[step * (index - 1):step * index]

            new_num = len(new_dataset)

            data_files.append((
                cls._get_dataset_file(new_dataset[:int(new_num * ratio)], output_dir,
                                      data_types[0], index, data_format),
                cls._get_dataset_file(new_dataset[int(new_num * ratio):], output_dir,
                                      data_types[1], index, data_format)))

            index += 1

        return data_files
                                  
from PIL import Image  # Import this to handle image files

def _hard_example_splitting(self, data_file, data_format, ratio,
                            data_types=None, output_dir=None, times=1):
    """
    Perform hard example splitting for datasets, including image datasets.
    
    Parameters
    ----------
    data_file : str
        Path to the dataset file (can be images, CSV, etc.).
    data_format : str
        Format of the dataset (e.g., CSV, TXT, JSON, JPG).
    ratio : float
        Ratio to split the data.
    data_types : tuple, optional
        Types of datasets to be split into (e.g., ("train", "eval")).
    output_dir : str, optional
        Directory to save the split datasets.
    times : int, optional
        Number of times to split the data.

    Returns
    -------
    list
        List of tuples with split dataset files.
    """
    if not data_types:
        data_types = ("train", "eval")

    if not output_dir:
        output_dir = tempfile.mkdtemp()

    all_data = self._read_data_file(data_file, data_format)

    data_files = []
    all_num = len(all_data)
    step = int(all_num / (times * 2))

    # First split: split the first half of the dataset
    data_files.append((
        self._get_dataset_file(all_data[:int((all_num * ratio) / 2)], output_dir,
                               data_types[0], 0, data_format),
        self._get_dataset_file(all_data[int((all_num * ratio) / 2):int(all_num / 2)], output_dir,
                               data_types[1], 0, data_format)
    ))

    # Subsequent splits
    index = 1
    while index <= times:
        if index == times:
            new_dataset = all_data[int(all_num / 2) + step * (index - 1):]
        else:
            new_dataset = all_data[int(all_num / 2) + step * (index - 1): int(all_num / 2) + step * index]

        new_num = len(new_dataset)

        data_files.append((
            self._get_dataset_file(new_dataset[:int(new_num * ratio)], output_dir,
                                   data_types[0], index, data_format),
            self._get_dataset_file(new_dataset[int(new_num * ratio):], output_dir,
                                   data_types[1], index, data_format)
        ))

        index += 1

    return data_files


@classmethod
def load_data(cls, file: str, data_type: str, label=None, use_raw=False, feature_process=None):
    """
    Load data from various formats including text, CSV, JSON, and image files (.jpg).
    
    Parameters
    ----------
    file: str
        The path to the data file (CSV, TXT, JSON, JPG).
    data_type: str
        The type of the data to be loaded for specific tasks.
    label: str, optional
        Label to be used for supervised learning datasets.
    use_raw: bool, optional
        Whether to use the raw data directly.
    feature_process: function, optional
        A function for additional feature processing on the raw data.

    Returns
    -------
    data_instance
        Parsed data instance, could be TxtDataParse, CSVDataParse, or image data.
    """
    data_format = utils.get_file_format(file)
    data = None

    if data_format == DatasetFormat.CSV.value:
        data = CSVDataParse(data_type=data_type, func=feature_process)
        data.parse(file, label=label)

    elif data_format == DatasetFormat.TXT.value:
        data = TxtDataParse(data_type=data_type, func=feature_process)
        data.parse(file, use_raw=use_raw)

    elif data_format == DatasetFormat.JSON.value:
        data = JSONDataParse(data_type=data_type, func=feature_process)
        data.parse(file)

    elif data_format in ['jpg', 'jpeg', 'png']:  # Handling image files
        data = Image.open(file)
        if feature_process:
            data = feature_process(data)

    return data
