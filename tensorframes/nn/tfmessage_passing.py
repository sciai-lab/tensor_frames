from typing import Any, Dict

from torch_geometric.nn import MessagePassing

from tensorframes.lframes.lframes import ChangeOfLFrames


class TFMessagePassing(MessagePassing):
    """TFMessagePassing class represents a message passing algorithm in the tensorframes formalism.

    https://arxiv.org/abs/2405.15389v1
    """

    def __init__(self, params_dict: Dict[str, Dict[str, Any]]) -> None:
        """Initializes a new instance of the TFMessagePassing class.

        Args:
            params_dict (Dict[str, Dict[str, Any]]): A dictionary containing the parameters for the message passing algorithm and the corresponding representations. Params which are not listed here are not transformed E.g.:
            {
                "feat_0": {
                    "type": "local"
                    "rep": TensorReps("1x0n")
                },
                "feat_1": {
                    "type": "global"
                    "rep": Irreps("1x0n")
                },
            }
        """
        self.params_dict = params_dict

        for key, value in self.params_dict.items():
            if value["type"] is not None:
                self.params_dict[key]["transform"] = value["rep"].get_transform_class()

        super().__init__()

        # Register hooks to call before propagating and before sending messages
        self.register_propagate_forward_pre_hook(self.pre_propagate_hook)
        self.register_message_forward_pre_hook(self.pre_message_hook)

    def pre_propagate_hook(self, module: Any, inputs: tuple) -> tuple:
        """A hook method called before propagating messages in the message passing algorithm. We
        save the lframes in the class variable and remove it from the inputs dictionary.

        Args:
            module (Any): The module object.
            inputs (Dict[str, Any]): The inputs dictionary.

        Returns:
            Dict[str, Any]: The modified inputs dictionary.
        """
        assert inputs[-1].get("lframes") is not None, "lframes are not in the propagate inputs"

        self.lframes = inputs[-1].pop("lframes")
        self.edge_index = inputs[0]

        return inputs

    def pre_message_hook(self, module: Any, inputs: tuple) -> tuple:
        """Pre-message hook method that is called before passing messages in the message passing
        algorithm. We transform the features according to the representations in the params_dict.

        Args:
            module (Any): The module object.
            inputs (Dict[str, Any]): The inputs dictionary.

        Returns:
            Dict[str, Any]: The modified inputs dictionary.
        """

        # calculate lframes_i, lframes_j and the U matrix
        lframes_i = self.lframes.index_select(self.edge_index[1])
        lframes_j = self.lframes.index_select(self.edge_index[0])
        U = ChangeOfLFrames(lframes_i, lframes_j)

        # now go through the params_dict and get the representations and transform the features in the right way
        for key, value in self.params_dict.items():
            if value["type"] == "local":
                assert inputs[-1].get(key + "_j") is not None, f"Key {key}_j not in inputs"
                # transform the features according to the representation
                inputs[-1][key + "_j"] = value["transform"](inputs[-1][key + "_j"], U)
            elif value["type"] == "global":
                assert inputs[-1].get(key) is not None, f"Key {key} not in inputs"
                # get the representation and apply it to the features
                inputs[-1][key] = value["transform"](inputs[-1][key], lframes_j)

        return inputs
