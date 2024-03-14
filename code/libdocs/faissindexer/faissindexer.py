import logging as log
import os
from enum import IntEnum
from os import path

import faiss
import numpy as np
import torch

log.basicConfig(level=log.CRITICAL)


class FaissIndexType(IntEnum):
    """
    FaissIndexType is a type that specifies the type of the Faiss index.
    """

    brute_force = 1
    hnsw = 2
    ivf = 3


class FaissIndexer:
    """
    FaissIndexer class wraps the Faiss library and keeps track of the index.
    It can be used to add vectors to the index, and ensure the updated index is persisted to disk.
    """

    def __init__(
        self,
        dimension: int,
        index_file: str = None,
        index_type: FaissIndexType = FaissIndexType.brute_force,
        metric_type: int = faiss.METRIC_L2,
        nlist: int = 100,
    ):
        """
        Constructor for the FaissIndexer class.

        :param dimension: the dimension of the vectors
        :param index_file: the index_file. if this is set to None, then create a new index
        :param index_type: the index_type. this must be set if the index_file is not provided
        :param metric_type: the type of metric to use for computing distances between vectors
        :param nlist: the number of cells in the quantizer
        """
        self.gpu = torch.cuda.is_available()
        if index_file is not None and path.isfile(index_file):
            log.info(f"loading existing faiss-index from file: {index_file}")
            self.index = self.load_index(index_file)
            log.info(
                f"index loaded with dimension: {self.index.d} and size: {self.index.ntotal}"
            )
        else:
            log.info(f"creating new faiss-index of type: {index_type}")
            if index_type == FaissIndexType.brute_force:
                self.index = self.__create_brute_force_index(
                    dimension=dimension
                )
            elif index_type == FaissIndexType.hnsw:
                self.index = self.__create_hnsw_index(dimension=dimension)
            elif index_type == FaissIndexType.ivf:
                self.index = self.__create_ivf_index(
                    dimension=dimension, metric_type=metric_type, nlist=nlist
                )
            else:
                log.error(
                    "invalid index type. valid types are: brute_force, hnsw, ivf."
                )
                raise ValueError(
                    "invalid index type. valid types are: brute_force, hnsw, ivf."
                )

            log.info(
                f"index created with dimension: {self.index.d} and size: {self.index.ntotal}"
            )
            # wrap up with IndexIDMap
            self.bare_index = self.index
            self.index = faiss.IndexIDMap(self.index)

            if self.gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    raise Exception(
                        f"error while moving faiss-index to GPU. Error: {e}"
                    )

        # ensure the dimensions
        self.dimension = dimension
        if dimension != self.index.d:
            raise ValueError(f"invalid dimension {dimension}/{self.index.d}")

    def __create_brute_force_index(self, dimension: int):
        """
        Create a brute-force index (slow for large datasets).

        :param dimension: the dimension of the vectors
        """
        log.info(f"creating brute-force index with dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        log.info(
            f"brute-force index created with dimension: {dimension} and size: {index.ntotal}"
        )
        return index

    def __create_hnsw_index(
        self, dimension: int, M: int = 16, efConstruction: int = 40
    ):
        """
        Create a HNSW index. This index is much faster than the brute-force index; however,
        it is not as accurate as the brute-force index.

        :param dimension: the dimension of the vectors
        :param M: the number of neighbors to explore at each query point
        :param efConstruction: the number of neighbors to index at construction time
        """
        log.info(
            f"creating HNSW index with dimension: {dimension}, M: {M}, efConstruction: {efConstruction}"
        )
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = efConstruction
        log.info(
            f"hnsw index created with dimension: {dimension} and size: {index.ntotal}"
        )
        return index

    def __create_ivf_index(
        self,
        dimension: int,
        metric_type: int = faiss.METRIC_L2,
        nlist: int = 100,
    ):
        """
        Create an IVF index. This index is much faster than the brute-force index;
        however, it needs to be trained with a set of vectors before it can be used for search.

        :param dimension: the dimension of the vectors
        :param metric_type: the type of metric to use for computing distances between vectors
        :param nlist: the number of cells in the quantizer
        """

        log.info(
            f"creating IVF index with dimension: {dimension}, metric_type: {metric_type}, nlist: {nlist}"
        )
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
        log.info(
            f"ivf index created with dimension: {dimension} and size: {index.ntotal}"
        )
        return index

    def load_index(self, index_file: str):
        """
        Load an existing index from file.

        :param index_file: the name of the file to load the index from.
        """
        log.info(f"Loading existing faiss-index from file: {index_file}")
        if not path.isfile(index_file):
            raise Exception(f"file does not exist: {index_file}")

        try:
            index = faiss.read_index(index_file)
        except Exception as e:
            raise Exception(
                f"error while loading faiss-index from file: {index_file}. Error: {e}"
            )

        if self.gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                raise Exception(
                    f"error while moving faiss-index to GPU. Error: {e}"
                )

        log.info(f"Index loaded from {index_file}")
        return index

    def save_index(self, index_file: str):
        """
        Save the index to file, creating the directory if it doesn't exist.

        Args:
            index_file (str): The path to the file where the index will be saved.
        """

        log.info(f"Saving Faiss-index to file: {index_file}")

        # Extract directory path and filename
        directory = os.path.dirname(index_file)
        os.makedirs(directory, exist_ok=True)

        try:
            if self.gpu:
                self.index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(self.index, index_file)
            if self.gpu:
                self.index = faiss.index_cpu_to_gpu(self.index)
        except Exception as err:
            log.error(f"error writing faiss index {err}")
            return err

    def size(self):
        return self.index.ntotal

    def add(self, embeddings, ids):
        if len(embeddings) != len(ids):
            raise ValueError(
                f"mismatched embeddings and ids. {len(embeddings)}/{len(ids)}"
            )
        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"mismatched shapes. {embeddings.shape[1]}/{self.index.d}"
            )

        log.info(
            f"adding {len(embeddings)}/{len(ids)} vectors to the index with dimension {self.index.d}"
        )
        self.index.add_with_ids(embeddings, np.array(ids))
