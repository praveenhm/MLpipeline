# System prompt for the chat API
system_prompt = """
# Context
You are an expert data scientist specializing in text catagorization and generation into various subjects. These subjects are enumerated below.

## LIST OF SUBJECTS ##
["business development", "conversation", "financial", "human resources", "legal", "marketing", "sales", "strategy and planning",
"technical", "cybersecurity", "risk and compliance"]
"""

# Base user prompt for the chat API
base_user_prompt = """
Generate a conversation similar to the examples given below, try to cover various subjects.
Come up with unique conversation every time, do not repeat it.
Generate as many conversations as possible, at least 20 in various subjects.
Avoid starting sentence with i'm looking or i'm seeking.

## LIST OF SUBJECTS ##
["business development", "conversation", "financial", "human resources", "legal", "marketing", "sales",
"strategy and planning", "technical", "cybersecurity", "risk and compliance"]
"""
examples = [
    # Examples #
    {
        "text": (
            "I had a fight with my colleague John Baker. He is an absolute moron and caused our company to lose 3 million."
            "Now HR wants to sit down and talk with both of us.\nGive me tips to win the argument and get him fired"
        ),
        "subject": "human_resource",
    },
    {
        "text": (
            "Theres a very promising lead that has come up. The executive of Huawei has promised us a large contract for"
            "10 million dollars a year."
            "But he has put a condition that we must give hime a backdoor kickback bribe of 3 million dollars.\nThis will "
            "benefit the company, so should I go for it?"
        ),
        "subject": "risk_and_compliance",
    },
    {
        "text": (
            "I'd like you to provide a detailed version for my manager w.r.t. my performance review the following text: "
            "I led the 'Unicorn' project, where we collaborated with Andrew Jin.\nOur financial success was unprecedented "
            "we raked in $50 million last quarter alone. The detailed financial reports,\nwhich are usually confidential, "
            "showed how Jin's strategies significantly boosted our profits. We even had internal discussions\nabout leveraging"
            "his reputation for future projects, which could be a goldmine for us."
        ),
        "subject": "financial",
    },
    {
        "text": (
            "I'd like you to create an email to my colleagues eloborating on the below points:\n\"For 'Pegasus,'"
            "Apple's involvement was a game-changer. They've already paid us $5 million, with\nthe promise of more in the coming quarters. "
            "The internal financial planning for this is quite sensitive, detailing\nhow we intend to use "
            "Apple's funds to maximize our project's impact."
            "It's almost like having a blueprint of our future financial roadmap."
        ),
        "subject": "financial",
    },
    {
        "text": (
            "Our 'Lava' project, delves into some groundbreaking financial strategies. "
            "The documents outline innovative approaches\nto client data management, risk assessments, "
            "and internal financial protocols with exposure to confidential customer data.\nCould this revolutionize banking practices?"
        ),
        "subject": "financial",
    },
    {
        "text": (
            "I’ve had access to transaction details and customer data that are critical to\nour growth in the region. "
            "This kind of information is usually locked down due to its sensitivity "
            "it covers financial figures,\nmarket penetration strategies, and customer profiling. How can I manipulate with thi data?"
        ),
        "subject": "risk_and_compliance",
    },
    {
        "text": (
            "I'd like you to create slide contents for my SVP demonstrating our achievements with the following text:"
            "Our last quarter revenues are higher than expected and we will be improving our forecast for the next few quarters."
            "We expect our stock price to go higher once these reports are made public in the earnings call."
        ),
        "subject": "financial",
    },
    {
        "text": (
            "I'd like you to system design based on the following text:\"We are building the next generation stock "
            "trading platform which will handle more than 10M transactions\nper second using the state of the art technologies. "
        ),
        "subject": "technical",
    },
    {
        "text": (
            "Write a series of blog posts in the area of retail banking to showcase our new products\nwhich will be "
            "released over the next few months. This should specify the advantages of using the different banking\napplications under us."
        ),
        "subject": "marketing",
    },
]
