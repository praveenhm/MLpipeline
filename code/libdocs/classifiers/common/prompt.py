subject_taxonomy_classifier_prompt = """You are responsible for classifiying some input into one of the following topics:

    - business development
    - conversation
    - financial
    - human resource
    - legal
    - marketing
    - sales
    - strategy and planning
    - technical
    - cybersecurity
    - risk and compliance

Your answer MUST be only a single topic you find the most relevant. If no topic is matching
you must solely reply:

    irrelevant

EXAMPLES:

INPUT: We began our risk management research back in 2007. This was the time when most large non-financial corporations were just starting to build risk management functions and implementing risk management frameworks. At the time, our study showed that risk management was largely driven by the stock exchange requirements and was very basic in nature.

We identified several challenges relating to weak risk culture and confusion around the roles and responsibilities that the boards, executives and the risk management teams play in the overall management of the company’s risks. We also noted that back in 2007, risk managers focused primarily on foundation activities, such as developing risk management frameworks, conducting basic  risk  assessments  and  preparing  risk  reports  that  did  not  show  a  clear  link  between identified risks and corporate objectives. This resulted in very compliance-like and sometimes overly bureaucratic procedures. It often took months to get any meaningful results and it quickly became  a box-ticking  exercise."
OUTPUT: risk-and-compliance


INPUT: After the interview finishes, she asks you to take a quick cognitive test, which you feel good about. She tells you she will be doing reference checks and will let you know by early next week. To  get  to  this  point,  the  hiring  manager  may  have  reviewed  hundreds  of  résumés  and  developed  criteria  she would use for selection of the right person for the job. She has probably planned a time line for hiring, developed hiring criteria, determined a compensation package for the job, and enlisted help of other managers to interview
candidates.  She  may  have  even  performed  a  number  of  phone  interviews  before  bringing  only  a  few  of  the best candidates in for interviews. It is likely she has certain qualities in mind that she is hoping you or another candidate will possess.
OUTPUT: human-resources


INPUT: In object-oriented programming, the goal is to encapsulate internal details within a class. Objects exchange messages through the public methods only, thus leading to loose coupling between components
OUTPUT: technical

INPUT: zm observations randomly using the bootstrap method [12]. Using Ŵ₂, 6₁, 62, and abootstrap sample of size m, we estimated W₂W₁ as follows: (n₂¹ (ŷ¡) – (Ŵ₂ɓ1 + ĥ2))x (x¡x )¯¹, where we estimate the inverse of xix in Equation (3) using the naive approach fromthe diagonal elements in xx. Additionally, using the generalized inverse approach,we obtained W₁ in the basis of Ŵ₂ and Ŵ₂Ŵ₁. Finally, 6₁, 62, Ŵ₁, and Ŵ₂ wereused as initial vectors and matrices to update the parameters of the convolutionalneural network.
OUTPUT: irrelevant

INPUT: TC 45(3): 367-373 (March 1996). 68. R. Muntz, and J. C. S. Lui. Performance analysis of disk arrays under failure.
OUTPUT: irrelevant

INPUT: Comput., 2010, 28, (1), pp. [ 51] Nanni, L., Maio, D.: 'Weighted sub-Gabor for face recognition', Pattern Recognit.
OUTPUT: irrelevant

INPUT: TABLE 5X7.2 Role of Different Components of Cluster Management Systems in Dealingwith Resource Contentions Walters et al. [ 109]Scojo-PECT [92] Cluster reserves [10]Muse [23] Provisioning OperationalModel Local sched andAdmission ctrl TABLE 57.3 Role of Different Components of Grid/Cloud Resource Management Systems in Dealingwith Resource Contentions Provisioning OperationalModel CloudCloudGridDesktop Grid ProactiveProactive Grid FederationOriginReactive Cloud Federation RequestReactiveReactive Cloud Federation RequestGridRequestGrid Federation InterdomainOrigin Reactive Desktop GridNOWGrid Federation PartitioningEconomic (utility) Partial preemptionGlobal schedulingPreemption Interdomain Global schedulingand Request and outsourcing Global schedAdmission ctrlOutsourcingAdmission ctrlGlobal schedAdmission ctrl Shirako [48] is a lease-based platform for on-demand allocation of resources across several Clusters.
OUTPUT: irrelevant

INPUT: Ibid., 230-232. 122. Sally Roberts, \"FMLA's Effects Weighed; Employ- ers Cite Paperwork as Onus,\" Business Insurance(January 15, 2001), 3;
OUTPUT: irrelevant

Proceed:

INPUT: {input}
OUTPUT:
"""


junk_classifier_prompt = """
You are an expert data scientist specializing in text clean up.

The user has given you a chunk of text that has been extracted from a textbook. Your task is to classify it as either junk or clean, no other output must be returned, not even an empty string. Return the result as given by the example:
OUTPUT: junk

In a textbook there can be extraneous information, and in order to identify them as clean or junk, you need to focus on what to classify as junk. As junk we consider any chunk of text that contains one of the following information:

* copyright information
* biblographic reference
* publication dates
* appendix data
* chunks that represent chapter headings or titles
* chunks where the majority of the text is a quote or a citation
* table headers
* represents just a piece of a glossary

In each of these cases, categorize the chunk as "junk", as given in the examples output below:

OUTPUT: junk

# Examples #

----

INPUT: We began our risk management research back in 2007. This was the time when most large non-financial corporations were just starting to build risk management functions and implementing risk management frameworks. At the time, our study showed that risk management was largely driven by the stock exchange requirements and was very basic in nature.
OUTPUT: clean

INPUT: After the interview finishes, she asks you to take a quick cognitive test, which you feel good about. She tells you she will be doing reference checks and will let you know by early next week. To  get  to  this  point,  the  hiring  manager  may  have  reviewed  hundreds  of  résumés  and  developed  criteria  she would use for selection of the right person for the job. She has probably planned a time line for hiring, developed hiring criteria, determined a compensation package for the job, and enlisted help of other managers to interview
candidates.  She  may  have  even  performed  a  number  of  phone  interviews  before  bringing  only  a  few  of  the best candidates in for interviews. It is likely she has certain qualities in mind that she is hoping you or another candidate will possess.
OUTPUT: clean

INPUT: In object-oriented programming, the goal is to encapsulate internal details within a class. Objects exchange messages through the public methods only, thus leading to loose coupling between components
OUTPUT: clean

INPUT: TC 45(3): 367-373 (March 1996). 68. R. Muntz, and J. C. S. Lui. Performance analysis of disk arrays under failure.
OUTPUT: junk

INPUT: Comput., 2010, 28, (1), pp. [ 51] Nanni, L., Maio, D.: 'Weighted sub-Gabor for face recognition', Pattern Recognit.
OUTPUT: junk

INPUT: TABLE 5X7.2 Role of Different Components of Cluster Management Systems in Dealingwith Resource Contentions Walters et al. [ 109]Scojo-PECT [92] Cluster reserves [10]Muse [23] Provisioning OperationalModel Local sched andAdmission ctrl TABLE 57.3 Role of Different Components of Grid/Cloud Resource Management Systems in Dealingwith Resource Contentions Provisioning OperationalModel CloudCloudGridDesktop Grid ProactiveProactive Grid FederationOriginReactive Cloud Federation RequestReactiveReactive Cloud Federation RequestGridRequestGrid Federation InterdomainOrigin Reactive Desktop GridNOWGrid Federation PartitioningEconomic (utility) Partial preemptionGlobal schedulingPreemption Interdomain Global schedulingand Request and outsourcing Global schedAdmission ctrlOutsourcingAdmission ctrlGlobal schedAdmission ctrl Shirako [48] is a lease-based platform for on-demand allocation of resources across several Clusters.
OUTPUT: junk

INPUT: Ibid., 230-232. 122. Sally Roberts, \"FMLA's Effects Weighed; Employ- ers Cite Paperwork as Onus,\" Business Insurance(January 15, 2001), 3;
OUTPUT: junk

INPUT: Chapter 13 Managing humanresources' [ISBN 9780195982169]. 1.1.6 Synopsis of chapter content In this chapter we define what is meant by HR management, explain whyHR policies programmes and plans are so important, and consider therelationship between HR management and productivity. We examine thedifference between the academic study of HR management and practice,and explain why theory is important.
OUTPUT: junk

INPUT: 1. Carreras, E., Alloza, A., & Carreras, A. (2 The economy of intangiblesand reputation. In Corporate reputation (pp.
OUTPUT: junk

INPUT: StrategicManagement Journal, 13(2), 135-144. 3. Suchman, M. (1995). Managing legitimacy: Strategic and institutionalapproaches. The Academy of Management Review, 20(3), 571-610.
OUTPUT: junk

INPUT: Marklin, R. and G. Simoneau (1996). Upper extremity posture of typists using alternative keyboards.
OUTPUT: junk

INPUT: Governments make classifications every day, so not all classifications can be illegalunder the equal protection clause. People with more income generally pay a greaterpercentage of their income in taxes. People with proper medical training arelicensed to become doctors; people without that training cannot be licensed andcommit a criminal offense if they do practice medicine.
OUTPUT: clean

INPUT: Cholakova, M., & Clarysse, B. (2015). Does the Possibility to Make EquityInvestments in Crowdfunding Projects Crowd Out Rewards-basedInvestments?
OUTPUT: junk

INPUT: 2019; Mochkabadi and Volkmann 2020) and even on the widersociety and environment (Testa et al. 2019; Vismara 2019). To entrepre-neurial ventures, equity crowdfunding offers an alternative form of equityfinancing that they may turn to out of choice or out of necessity (Walthoff-Borm et al. 2018).
OUTPUT: junk

INPUT: Dushnitsky, G., & Fitza, M. A. (2018). Are We Missing the Platforms for theCrowd?
OUTPUT: junk

INPUT: Yoon, Y., Li, Y., & Feng, Y. (2019). Factors Affecting Platform Default Risk inOnline Peer-to-Peer (P2P) Lending Business: An Empirical Study UsingChinese Online P2P Platform Data.
OUTPUT: junk

INPUT: Electronic Commerce Research,19(1), 131-158. Yum, H., Lee, B., & Chae, M. (2012). From the Wisdom of Crowds to MyOwn Judgment in Microfinance Through Online Peer-to-Peer LendingPlatforms.
OUTPUT: junk

INPUT: Agus, A., and M. Shukri Hajinoor. 2012. Lean Production Supply ChainManagement as Driver Towards Enhancing Product Quality and BusinessPerformance. International Journal of Quality & Reliability Management 29(1): 92–121.
OUTPUT: junk

INPUT: Guenther, C., Johan, S., & Schweizer, D. (2018). Is the Crowd Sensitive toDistance?
OUTPUT: junk

INPUT: The investigated logistics operation is cross-docking where the goodsmust be unloaded from trucks to the cross-dock area in DC. Today,DC operators manually pick up the goods by hand from the deliverytruck and put them on the vertical conveyor belt. Then the goods will (a) Heavy and Long Fabric Rolls (c) Packed Second handed Tyres Fig.
OUTPUT: clean

INPUT: The other group of companies has already voluntarily implemented stan-dards, perhaps even including an integrated reporting framework and audits. This second group of companies will have to consider how to deal with theiradvanced reporting formats in light of the new regulations, as the integrationof the sustainability report into the management report under the CSRDseems to be of limited scope compared to the Integrated Reporting Framework(cf.
OUTPUT: clean

INPUT: If relevant you might include information showing the financing required. Fixedcapital is usually financed using longer-term sources, like long-term loans. The best way to finance fixed capital requirements is often by attempting to match thefinancing term with the length of time the asset will be used by the company. Fixed assetsgenerally last for a longer period of time, so they should be financed with longer-term loans orequity financing.
OUTPUT: clean

INPUT: Bergson, H. (1913/2001). Time and free will: An essay on the immediate data ofconsciousness [Kindle version].
OUTPUT: junk

Proceed:

INPUT: {input}
OUTPUT:
"""
