from libdocs.utils.label.label import split


def test_clean_label():

    test_label_lists = [
        {
            "test": ["_hello"],
            "expected": ["hello"],
        },
        {
            "test": ["a"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["e"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["1"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["the"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["an"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["_hello_"],
            "expected": ["hello"],
        },
        {
            "test": ["__hello__"],
            "expected": ["hello"],
        },
        {
            "test": ["__hello:__"],
            "expected": ["hello"],
        },
        {
            "test": ["__hello_or_goodbye__(..xxx)"],
            "expected": ["hello", "goodbye"],
        },
        {
            "test": [">_library"],
            "expected": ["library"],
        },
        {
            "test": [">_library_<"],
            "expected": ["library"],
        },
        {
            "test": ['">_library_<"'],
            "expected": ["library"],
        },
        {
            "test": ["'>_library_<'"],
            "expected": ["library"],
        },
        {
            "test": ["pharmaceutical_(a_subcategory_of_healthcare)"],
            "expected": ["pharmaceutical"],
        },
        {
            "test": ["statistics"],
            "expected": ["statistics"],
        },
        {
            "test": [
                "It_is_not_clear_enough_to_classify_as_a_specific_topic._However,_based_on_the_context,_it_seems_to_be_related_to_risk_management_and_compliance._Therefore,_I_would_suggest_the_topic_as_risk_and_compliance."  # noqa: E501
            ],
            "expected": ["irrelevant"],
        },
        {
            "test": ["insurance"],
            "expected": ["insurance"],
        },
        {
            "test": ["relevant_(to_the_topic_of_wikipedia)"],
            "expected": ["relevant"],
        },
        {
            "test": [
                "advantage:_scripting_languages_are_interpreted,_which_makes_them_easier_to_learn_and_use_than_compiled_languages._Disadvantage:_because_they_are_interpreted,_scripting_languages_can_be_slower_in_execution_than_compiled_languages."  # noqa: E501
            ],
            "expected": ["advantage"],
        },
        {
            "test": ["education_related"],
            "expected": ["education_related"],
        },
        {
            "test": ["performance_evaluation"],
            "expected": ["performance_evaluation"],
        },
        {
            "test": ["confidentiality_agreements"],
            "expected": ["confidentiality_agreements"],
        },
        {
            "test": ["founder"],
            "expected": ["founder"],
        },
        {
            "test": ["security_and_essential_interests"],
            "expected": ["irrelevant"],
        },
        {
            "test": ["patent"],
            "expected": ["patent"],
        },
        {
            "test": ["programming"],
            "expected": ["programming"],
        },
        {
            "test": ["administrative_burden"],
            "expected": ["administrative_burden"],
        },
        {
            "test": ["history_of_management"],
            "expected": ["history_of_management"],
        },
        {
            "test": ["psychological_well_being"],
            "expected": ["psychological_well_being"],
        },
        {
            "test": ["interviewing_and_hiring"],
            "expected": ["interviewing_and_hiring"],
        },
        {
            "test": ["labor_relations"],
            "expected": ["labor_relations"],
        },
        {
            "test": ["philosophy"],
            "expected": ["philosophy"],
        },
        {
            "test": [
                "telecommunications_(technical_or_business_development,_depending_on_the_context)"
            ],
            "expected": ["telecommunications"],
        },
        {
            "test": ["consulting"],
            "expected": ["consulting"],
        },
        {
            "test": ["socio_technical"],
            "expected": ["socio_technical"],
        },
        {
            "test": ["creative_arts"],
            "expected": ["creative_arts"],
        },
        {
            "test": ["political_activism"],
            "expected": ["political_activism"],
        },
        {
            "test": [
                "health_care_or_business_development_(depending_on_the_context_of_the_organization)"
            ],
            "expected": ["health_care", "business_development"],
        },
        {
            "test": ["safety_and_compliance_(risk_and_compliance)"],
            "expected": ["safety_and_compliance"],
        },
        {
            "test": [""],
            "expected": ["irrelevant"],
        },
        {
            "test": ["|_relevant"],
            "expected": ["relevant"],
        },
        {
            "test": ["\u25a1_relevant"],
            "expected": ["relevant"],
        },
    ]

    for label_list in test_label_lists:
        labels = split(label_list["test"])
        assert len(labels) == len(label_list["expected"]), f"{label_list}"
        for index, label in enumerate(labels):
            assert (
                label == label_list["expected"][index]
            ), f"{label_list}: {index}: {labels}: {label_list['expected']}"
