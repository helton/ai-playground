from abc import abstractmethod
from typing import Optional, List

from llama_index import QueryBundle
from llama_index.schema import NodeWithScore


class DuplicateRemoverNodePostProcessor:
    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        print("_postprocess_nodes enter")
        unique_hashes = set()
        unique_nodes = []
        for node in nodes:
            node_hash = node.node.hash
            if node_hash not in unique_hashes:
                unique_hashes.add(node_hash)
                unique_nodes.append(node)
        return unique_nodes
