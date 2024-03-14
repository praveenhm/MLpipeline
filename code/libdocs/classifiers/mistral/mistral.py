from libdocs.classifiers.common.common import LLMCommon


class MistralInstruct(LLMCommon):

    def __init__(self, sampling_params=None):
        super(MistralInstruct, self).__init__(
            "mistralai/Mistral-7B-Instruct-v0.2", sampling_params
        )
