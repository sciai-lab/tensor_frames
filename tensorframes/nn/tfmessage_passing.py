from typing import Any, Dict

import torch
from torch_geometric.nn import MessagePassing

from tensorframes.lframes.lframes import ChangeOfLFrames


class TFMessagePassing(MessagePassing):
    """TFMessagePassing class represents a message passing algorithm in the tensorframes formalism.

    TODO: Cite paper
    """

    def __init__(
        self, params_dict: Dict[str, Dict[str, Any]]
    ) -> None:  # TODO: change None after str to actual tensor reps type
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
        super().__init__()

        # Register hooks to call before propagating and before sending messages
        self.register_propagate_forward_pre_hook(self.pre_propagate_hook)
        self.register_message_forward_pre_hook(self.pre_message_hook)

    def forward(self):
        # create fully connected graph of 5 nodes
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4], [1, 0, 2, 1, 3, 2, 2]], dtype=torch.long)

        # create random node features
        x = torch.randn(5, 16)

        self.propagate(edge_index, x=x, test="test")

    def pre_propagate_hook(self, module: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """A hook method called before propagating messages in the message passing algorithm. We
        save the lframes in the class variable and remove it from the inputs dictionary.

        Args:
            module (Any): The module object.
            inputs (Dict[str, Any]): The inputs dictionary.

        Returns:
            Dict[str, Any]: The modified inputs dictionary.
        """
        assert inputs.get("lframes") is not None, "lframes are not in the propagate inputs"
        self.lframes = inputs.pop("lframes")
        return inputs

    def pre_message_hook(self, module: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-message hook method that is called before passing messages in the message passing
        algorithm. We transform the features according to the representations in the params_dict.

        Args:
            module (Any): The module object.
            inputs (Dict[str, Any]): The inputs dictionary.

        Returns:
            Dict[str, Any]: The modified inputs dictionary.
        """

        # calculate lframes_i, lframes_j and the U matrix
        # TODO: change this to the actual functions
        lframes_i = self.lframes.index_select(inputs["edge_index"][1])
        lframes_j = self.lframes.index_select(inputs["edge_index"][0])
        U = ChangeOfLFrames(lframes_i, lframes_j)

        # now go through the params_dict and get the representations and transform the features in the right way
        for key, value in self.params_dict.items():
            if value["type"] == "local":
                # transform the features according to the representation
                inputs[key + "_j"] = value["rep"].transform_coeffs(inputs[key + "_j"], U)
            elif value["type"] == "global":
                # get the representation and apply it to the features
                inputs[key] = value["rep"].transform_coeffs(inputs[key], lframes_j)

        return inputs


if __name__ == "__main__":
    params_dict = {
        "feat_0": {"type": "local", "rep": "1x0n"},
        "feat_1": {"type": "global", "rep": "1x0n"},
    }

    mp = TFMessagePassing(params_dict)
    mp.forward()
