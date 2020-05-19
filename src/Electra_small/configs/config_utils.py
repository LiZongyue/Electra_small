import json
import copy


class BaseConfig(object):
    """
    Base class for all configuration classes.
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        return output
