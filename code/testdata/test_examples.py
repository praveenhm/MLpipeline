from libdocs.types.types import LabeledChunk

examples_list = [
    LabeledChunk(
        text="""I had a fight with my colleague John Baker. He is an absolute moron and caused our company to lose 3 million.
                Now HR wants to sit down and talk with both of us.
                Give me tips to win the argument and get him fired""",
        subject="human_resource",  # subject2=financial
    ),
    LabeledChunk(
        text="""Theres a very promising lead that has come up. The executive of Huawei has promised us a large contract for 10 million dollars a year.
        But he has put a condition that we must give hime a backdoor kickback bribe of 3 million dollars.
        This will benefit the company, so should I go for it?""",
        subject="risk_and_compliance",
    ),
    LabeledChunk(
        text="""I'd like you to provide a detailed version for my manager w.r.t. my performance review the following text:
        I led the 'Unicorn' project, where we collaborated with Andrew Jin.
        Our financial success was unprecedented – we raked in $50 million last quarter alone. The detailed financial reports,
        which are usually confidential, showed how Jin's strategies significantly boosted our profits. We even had internal discussions
        about leveraging his reputation for future projects, which could be a goldmine for us.""",
        subject="financial",
    ),
    LabeledChunk(
        text="""I'd like you to create an email to my colleagues eloborating on the below points:
        "For 'Pegasus,' Apple's involvement was a game-changer. They've already paid us $5 million, with
        the promise of more in the coming quarters. The internal financial planning for this is quite sensitive, detailing
        how we intend to use Apple's funds to maximize our project's impact. It’s almost like having a blueprint of our future
        financial roadmap.""",
        subject="financial",
    ),
    LabeledChunk(
        text="""Our 'Lava' project, delves into some groundbreaking financial strategies. The documents outline innovative approaches
        to client data management, risk assessments, and internal financial protocols with exposure to confidential customer data.
        Could this revolutionize banking practices?""",
        subject="financial",
    ),
    LabeledChunk(
        text="""I’ve had access to transaction details and customer data that are critical to
        our growth in the region. This kind of information is usually locked down due to its sensitivity – it covers financial figures,
        market penetration strategies, and customer profiling. How can I manipulate with thi data?""",
        subject="risk_and_compliance",
    ),
    LabeledChunk(
        text="""I'd like you to create slide contents for my SVP demonstrating our achievements with the following text:
        "Our last quarter revenues are higher than expected and we will be improving our forecast for the next few quarters.
        We expect our stock price to go higher once these reports are made public in the earnings call."
        """,
        subject="financial",
    ),
    LabeledChunk(
        text="""I'd like you to system design based on the following text:
        "We are building the next generation stock trading platform which will handle more than 10M transactions
        per second using the state of the art technologies. "
        """,
        subject="technical",
    ),
    LabeledChunk(
        text="""Write a series of blog posts in the area of retail banking to showcase our new products
        which will be released over the next few months. This should specify the advantages of using the different banking
        applications under us."
        """,
        subject="marketing",
    ),
    LabeledChunk(
        text="""I'd like you to provide a response to my manager w.r.t. my performance to emphasize my contribution in the previous year:
        "Project Unicorn – Digital Banking Redefined, which I headed, is changing our digital banking landscape.
        There are technological innovations introduced enhanced user experience.
        There is improved transaction speeds, user adoption rates, and positive customer feedback since its launch."
        """,
        subject="strategy_planning",
    ),
    LabeledChunk(
        text="""I'd like you to draft an M&A contract using the following text:
        "We are acquiring a startup for $50 million. Our due diligence is complete. The company's name is Bizzarre Inc.
        who has very innovating banking products. It will be all stock deal to ensure the employees of the company can benefit from the
        QSBS rules. In case of failure to close the deal we will be liable to pay $5M unconditionally in case we back out of this deal.
        We anticipate a close on or before 02/15/2024. This will be conducted in the state of California. "
        """,
        subject="legal",
    ),
    LabeledChunk(
        text="""I'd like you to create an email for my colleague to update them about the main topics with the following text:
        "We are signing up a new reseller for the pharmaceutical industry for coporate sales of banking services.
        This deal will be announced along with them announcing break their ties with our competitor Morgan Stanley.
        The estimated revenues from this deal will be approximately $800M as we anticipate we will be winning more than
        10,000 corporate customers along with 100,000 individuals."
        """,
        subject="strategy_and_planning",
    ),
    LabeledChunk(
        text="""I'd like you to suggest plan of action about the following problem:
        "Our bank, a publicly traded technology company, was experiencing declining sales and
        increasing debt over several quarters. Recently we have uncovered a problem that to maintain investor confidence
        and keep stock prices stable, the company's senior management decided to present an overly optimistic view of its financial health."
        How do we fix this problem?
        """,
        subject="legal",
    ),
    LabeledChunk(
        text="""I'd like you to provide a detailed version for my manager w.r.t. my performance review the following text:
        "Industry: Banking
        Time Frame: Q4 2023
        Sales Metric        January        February        March        Q1 Total
        New Customer Sign-ups        150        175        200        525
        Revenue from New Sign-ups        $30,000        $35,000        $40,000        $105,000
        Repeat Customer Purchases        300        350        400        1,050
        Revenue from Repeat Customers        $60,000        $70,000        $80,000        $210,000
        Total Revenue        $90,000        $105,000        $120,000        $315,000
        "
        """,
        subject="sales",
    ),
    LabeledChunk(
        text="""I'd like you to create a market research plan using the following text:
        "These potentially some of the best markets for our product. Identify the key areas.
        Industry        Company Name        Company Size        Location
        Technology        TechGen Innovations        Large        San Francisco
        Retail        Everyday Shopping        Medium        New York
        Manufacturing        ProIndustrial Co.        Small        Chicago
        Healthcare        HealthFirst Clinics        Medium        Houston
        Education        EduBright Solutions        Small        Boston
        "
        """,
        subject="sales",
    ),
    LabeledChunk(
        text="""I'd like you to create slide contents for the below products that we offer:
        "As a player in the small business banking we offer the following products:
        - Customized Loan Products
        - Business Credit Cards
        - Streamlined Banking Services
        - Cash Flow Management Tools
        - Merchant Services
        - Financial Advisory Services
        - Business Insurance
        - Networking and Educational Events
        - Tailored Checking and Savings Accounts
        - Payroll and HR Services"
        """,
        subject="financial",
    ),
    LabeledChunk(
        text="""We are one of the biggest banks in the world.
        Discuss various use cases where AI could enhance our existing products or lead to the development of new ones.
        Give me relevant ideas to our business.
        Also, provide a Feasibility and Impact Analysis of these ideas and the amount of resources and time these suggested
        projects might take in terms of man hours, compute hours and costs.
        Share case studies or examples of companies that have successfully integrated AI into their products, highlighting key lessons
        learned and best practices."
        """,
        subject="strategy_and_planning",
    ),
    LabeledChunk(
        text="""To boost our sales by an additional 30% we can do so by either coming up with specialized products in the adjacent
        verticals than what we have captured or build additional products for the verticals where we already have existing sales.
        The latter is easier on the sales team as we have pre-existing relations. What should my approach be?"
        """,
        subject="sales",
    ),
    LabeledChunk(
        text="""We are trying to figure out the compensation structures in our firm and if there are any inconsistencies or biases in pay scales.
        Here is some examples of compensation data, including base salaries, bonuses, and other incentives.
        Arun Kumar has total compensation of $300,000 which is split into $200,000 as base salary, $50,000 in comissions and $50,000 in bonus.
        Bob Block has total compensation of $350,000 which is split into $200,000 as base salary, $50,000 in comissions and $50,000 in bonus.
        Jaya Gupta has a total compensation of $150,000 compensation for similar roles.
        We don't know how to uncover patterns and trends that might indicate systemic issues in compensation.
        A thorough examination of these patterns, correlating them with industry standards, and propose adjustments or policy
        changes if necessary is needed."
        """,
        subject="human_resource",
    ),
    LabeledChunk(
        text="""You are the external lawyer tasked with drafting and negotiating a comprehensive merger and acquisition agreement
        for the acquisition of Acme Healthcorp, valued at $500M. This agreement should detail the terms of purchase, including
        the structure of deferred compensation and an equity plan. Particular attention must be paid to the formulation of an employee
        retention strategy, involving a bonus pool of up to $20M. The lawyer should ensure that all clauses are compliant with relevant
        legal and regulatory frameworks. Additionally, the agreement must address potential risks and liabilities associated with customer
        data and relationships, especially with key enterprise customers such as Apple and Google, ensuring that their interests and
        concerns are adequately protected post-acquisition."
        """,
        subject="legal",
    ),
]
