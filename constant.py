To update the `DatasetFormat` class to include image data formats like `jpg`, you can modify the enum as follows:

```python
class DatasetFormat(Enum):
    """
    File format of inputting dataset.
    Currently, file formats are as follows: txt, csv, json, jpg, png.
    """
    CSV = "csv"
    TXT = "txt"
    JSON = "json"
    JPG = "jpg"
    PNG = "png"
```

This change will ensure that image formats such as `jpg` and `png` are also supported for the input dataset.
