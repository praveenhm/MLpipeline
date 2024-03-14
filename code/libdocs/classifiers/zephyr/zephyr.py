from libdocs.classifiers.common.common import LLMCommon


class ZephyrBeta(LLMCommon):

    def __init__(self, sampling_params=None):
        super(ZephyrBeta, self).__init__(
            "HuggingFaceH4/zephyr-7b-beta", sampling_params
        )
