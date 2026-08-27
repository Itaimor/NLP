Balancing Methods for Multi-label Text Classification with Long-Tailed
|                               |                         |     | Class Distribution |                           |     |     |
| ----------------------------- | ----------------------- | --- | ------------------ | ------------------------- | --- | --- |
|                               | YiHuang                 |     |                    | BuseGiledereli            |     |     |
|                               | DataandAnalyticsChapter |     |                    | ComputerEngineering       |     |     |
| Roche(China)HoldingLtd.,China |                         |     |                    | BogaziciUniversity,Turkey |     |     |
|                               | yi.huang.yh4@roche.com  |     |                    | DataandAnalyticsChapter   |     |     |
RocheMüstahzarlarıSanA.S¸.,Turkey
busegiledereli@gmail.com
| AbdullatifKöksal |     |     | ArzucanÖzgür |     | ElifOzkirimli |     |
| ---------------- | --- | --- | ------------ | --- | ------------- | --- |
ComputerEngineering ComputerEngineering DataandAnalyticsChapter
BogaziciUniversity,Turkey BogaziciUniversity,Turkey F.Hoffmann-LaRocheAG
abdullatif.koksal@boun.edu.tr arzucan.ozgur@boun.edu.tr Switzerland
elif.ozkirimli@roche.com
Abstract
PENNCENTRAL<PC>
Title1
SELLSU.K.UNIT
| Multi-label | text classification | is  | a challenging |     |     |     |
| ----------- | ------------------- | --- | ------------- | --- | --- | --- |
acq(1650),strategic-metal(16)
| task because | it requires | capturing | label de- | Labels1 |     |     |
| ------------ | ----------- | --------- | --------- | ------- | --- | --- |
nickel(8)
| pendencies. | It becomes         | even more       | challeng- |     |     |     |
| ----------- | ------------------ | --------------- | --------- | --- | --- | --- |
| ing when    | class distribution | is long-tailed. | Re-       |     |     |     |
U.S.MINTSEEKINGOFFERS
| sampling | and re-weighting | are | common ap- | Title2 |     |     |
| -------- | ---------------- | --- | ---------- | ------ | --- | --- |
ONCOPPER,NICKEL
| proaches | used for addressing | the | class imbal- |     |     |     |
| -------- | ------------------- | --- | ------------ | --- | --- | --- |
ance problem, however, they are not effective Labels2 copper(47),nickel(8)
| when there | is label dependency |     | besides class |     |     |     |
| ---------- | ------------------- | --- | ------------- | --- | --- | --- |
imbalancebecausetheyresultinoversampling Figure 1: Samples for multi-label text classification
|                 |                        |     |     | taskfromReuters-21578dataset. |     | Onlytitlesareshown |
| --------------- | ---------------------- | --- | --- | ----------------------------- | --- | ------------------ |
| ofcommonlabels. | Here,weintroducetheap- |     |     |                               |     |                    |
plicationofbalancinglossfunctionsformulti- forillustration. Numbersafterthelabelsindicatetotal
label text classification. We perform experi- numberofoccurrencesinthedataset.
| ments on               | a general domain | dataset               | with 90 |     |     |     |
| ---------------------- | ---------------- | --------------------- | ------- | --- | --- | --- |
| labels (Reuters-21578) |                  | and a domain-specific |         |     |     |     |
fromtheReuters-21578multi-labeltextclassifica-
| dataset from | PubMed with | 18211 | labels. We |     |     |     |
| ------------ | ----------- | ----- | ---------- | --- | --- | --- |
findthatadistribution-balancedlossfunction, tion dataset (Hayes and Weinstein, 1990). Here,
| which inherently | addresses | both | the class im- |                  |                |              |
| ---------------- | --------- | ---- | ------------- | ---------------- | -------------- | ------------ |
|                  |           |      |               | for the document | with the title | PENN CENTRAL |
balance and label linkage problems, outper- <PC>SELLSU.K.UNIT,theaimistofindthela-
| forms commonly | used | loss functions. | Dis- |     |     |     |
| -------------- | ---- | --------------- | ---- | --- | --- | --- |
belsacq(acquisitions),strategic-metal,andnickel
| tribution | balancing methods | have | been suc- |     |     |     |
| --------- | ----------------- | ---- | --------- | --- | --- | --- |
from90labels.
| cessfully | used in the image | recognition | field. |     |     |     |
| --------- | ----------------- | ----------- | ------ | --- | --- | --- |
Here, we show their effectiveness in natural Multi-labelclassificationbecomescomplicated
language processing. Source code is avail- whenthereisalong-taileddistribution(classimbal-
able at https://github.com/blessu/ ance)andlinkage(co-occurrence)oflabels. Class
BalancedLossNLP.
imbalanceoccurswhenasmallsubsetofthelabels
(namelyheadlabels)havemanyinstances, while
1 Introduction
majorityofthelabels(namelytaillabels)haveonly
Multi-label text classification is one of the core a few instances. For example, half of the labels
topicsinnaturallanguageprocessing(NLP)andis intheReutersdataset,includingcopper,strategic-
usedinmanyapplicationssuchassearch(Prabhu metal, and nickel, occur in less than 5% of the
etal.,2018)andproductcategorization(Agrawal trainingdata. Labelco-occurrenceorlabellinkage
etal.,2013). Itaimstofindtherelatedlabelsfrom isachallengewhensomeheadlabelsco-occurwith
afixed-setoflabelsforagiventextthatmayhave rareortaillabels,resultinginbiasforclassification
multiple labels. Figure 1 demonstrates examples totheheadlabels. Forexample,eventhoughthela-
8153
Proceedingsofthe2021ConferenceonEmpiricalMethodsinNaturalLanguageProcessing,pages8153–8161
November7–11,2021.(cid:13)c2021AssociationforComputationalLinguistics

belnickeloccurslessfrequently,theco-occurrence
informationofnickel/copper,nickel/strategic-metal
| isimportantforaccuratemodeling(Figure1). |     |     |     |     |     | So- |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4000
| lutionssuchasresamplingofthesampleswithless- |     |     |     |     |     |     |     | selcitrA fo rebmuN |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- |
3000
frequentlabelsinclassification(Estabrooksetal.,
| 2004;Charteetal.,2015),usingco-occurrencein-   |     |     |     |     |     |     |     | 2000 |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
| formationinthemodelinitialization(Kurataetal., |     |     |     |     |     |     |     | 1000 |     |     |     |     |     |
| 2016),orprovidingahybridsolutionforheadand     |     |     |     |     |     |     |     | 0    |     |     |     |     |     |
tailcategorieswithamulti-taskarchitecture(Yang 0 510152025303540455055606570758085
Sorted Label Index
|                                          |     |     |     |     |     |     |     | 0   |     |     |     |     | 1.0 |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| etal.,2020)havebeenproposedinNLP,however |     |     |     |     |     |     |     | 5   |     |     |     |     |     |
01510252035304540555065607570858
theyarenotsuitableforimbalanceddatasetsorthey
0.8
aredependentonthemodelarchitecture.
xednI lebaL detroS
Multi-labelclassificationhasbeenwidelystud-
0.6
| ied in the | computer |     | vision | (CV) domain, |     | and re- |     |     |     |     |     |     |     |
| ---------- | -------- | --- | ------ | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
0.4
| cently  | has benefited |           | from cost-sensitive |            | learning |        |     |     |     |     |     |     |     |
| ------- | ------------- | --------- | ------------------- | ---------- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- |
| through | loss          | functions | for                 | tasks such | as       | object |     |     |     |     |     |     |     |
0.2
| recognition | (Durand  |              | et al., | 2019; Milletari |         | et al., |     |     |     |     |     |     |     |
| ----------- | -------- | ------------ | ------- | --------------- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- |
| 2016),      | semantic | segmentation |         | (Ge             | et al., | 2018),  |     |     |     |     |     |     |     |
0.0
0 510152025303540455055606570758085
and medical imaging (Li et al., 2020a). Balanc- Sorted Label Index
| ing loss | functions      | such | as   | focal loss | (Lin       | et al., |        |     |                 |              |     |     |           |
| -------- | -------------- | ---- | ---- | ---------- | ---------- | ------- | ------ | --- | --------------- | ------------ | --- | --- | --------- |
| 2017),   | class-balanced |      | loss | (Cui et    | al., 2019) | and     |        |     |                 |              |     |     |           |
|          |                |      |      |            |            |         | Figure | 2:  | The long-tailed | distribution |     | and | label co- |
distribution-balancedloss(Wuetal.,2020)provide
|     |     |     |     |     |     |     | occurrence |     | for the Reuters-21578 |     |     | dataset. | The co- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --------------------- | --- | --- | -------- | ------- |
improvementstoresolvetheclassimbalanceand
|               |          |          |              |             |             |      | occurence         | matrix      | is color | coded    | based | on      | the condi- |
| ------------- | -------- | -------- | ------------ | ----------- | ----------- | ---- | ----------------- | ----------- | -------- | -------- | ----- | ------- | ---------- |
| co-occurrence |          | problems | in           | multi-label | classifica- |      |                   |             |          |          |       |         |            |
|               |          |          |              |             |             |      | tional            | probability | p(i|j)   | of class | in    | the ith | column on  |
| tion in       | CV. Loss | function | manipulation |             | has         | also | classinthejthrow. |             |          |          |       |         |            |
beenexplored(Lietal.,2020b;Cohanetal.,2020)
inNLPasitworksinamodelarchitecture-agnostic
|     |     |     |     |     |     |     | {(x1,y1),...,(xN,yN)}withN |     |     |     | traininginstances, |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | ------------------ | --- | --- |
fashionbyexplicitlyembeddingthesolutioninto
yk
|                |     |              |     |           |         |     | each        | having | a multi-label |                           | ground | truth | of = |
| -------------- | --- | ------------ | --- | --------- | ------- | --- | ----------- | ------ | ------------- | ------------------------- | ------ | ----- | ---- |
| the objective. |     | For example, |     | Li et al. | (2020b) | has |             |        |               |                           |        |       |      |
|                |     |              |     |           |         |     | [yk,...,yk] |        | ∈ {0,1}C      | (C isthenumberofclasses), |        |       |      |
borroweddice-basedlossfunctionfromamedical
1 C
|                                               |             |     |              |     |          |       | andaclassifieroutputzk |     |                 | =   | [zk,...,zk] |      | ∈ R,BCE |
| --------------------------------------------- | ----------- | --- | ------------ | --- | -------- | ----- | ---------------------- | --- | --------------- | --- | ----------- | ---- | ------- |
| imagesegmentationtask(Milletarietal.,2016)and |             |     |              |     |          |       |                        |     |                 |     | 1           | C    |         |
|                                               |             |     |              |     |          |       | is defined             |     | as (the average |     | reduction   | step | is not  |
| reported                                      | significant |     | improvements |     | over the | stan- |                        |     |                 |     |             |      |         |
shownforsimplicity):
| dard cross-entropy                         |     |     | loss function | in  | several | NLP |     |     |            |     |      |     |     |
| ------------------------------------------ | --- | --- | ------------- | --- | ------- | --- | --- | --- | ---------- | --- | ---- | --- | --- |
| tasks.                                     |     |     |               |     |         |     |     |     | (cid:40)   |     |      |     |     |
|                                            |     |     |               |     |         |     |     |     | −log(pk)   |     | ifyk | =1  |     |
|                                            |     |     |               |     |         |     |     | L   | =          | i   |      | i   | (1) |
| Inthiswork,ourmajorcontributionistheintro- |     |     |               |     |         |     |     | BCE | −log(1−pk) |     |      |     |     |
i otherwise.
ductionoftheuseofbalancinglossfunctionstothe
Thesigmoidfunctionisusedforcomputingpk,
| NLPdomainforthemulti-labeltextclassification |     |     |     |     |     |     |     |        |     |     |     |     | i   |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
|                                              |     |     |     |     |     |     | pk  | σ(zk). |     |     |     |     |     |
task. WeperformexperimentsonReuters-21578,a = TheplainBCEisvulnerabletolabel
|     |     |     |     |     |     |     | i   | i   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
imbalanceduetothedominanceofheadclassesor
generalandsmalldataset,andPubMed,abiomed-
|                      |     |     |           |          |     |      | negativeinstances(Durandetal.,2019). |     |     |     |     |     | Below,we |
| -------------------- | --- | --- | --------- | -------- | --- | ---- | ------------------------------------ | --- | --- | --- | --- | --- | -------- |
| ical domain-specific |     |     | and large | dataset. | For | both |                                      |     |     |     |     |     |          |
datasets, the distribution balancing methods not describethreealternativeapproachesthataddress
theclassimbalanceprobleminlong-taileddatasets
onlyoutperformtheotherlossfunctionsfortheto-
|     |     |     |     |     |     |     | inmulti-labeltextclassification. |     |     |     | Themainideaof |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | ------------- | --- | --- |
talmetrics,butalsoleadtosignificantimprovement
thesebalancingmethodsistoreweightBCEsothat
| for the | tail labels. | We  | suggest | that | the balancing |     |     |     |     |     |     |     |     |
| ------- | ------------ | --- | ------- | ---- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
lossfunctionsprovidearobustsolutionforaddress- rareinstance-labelpairsintuitivelygetreasonable
“attention”.
ingthechallengesinmulti-labeltextclassification.
|     |     |     |     |     |     |     | 2.1 | Focalloss(FL) |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- |
2 LossFunctions
BymultiplyingamodulatingfactortoBCE(with
In NLP, Binary Cross Entropy (BCE) loss is thetunablefocusingparameterγ ≥ 0),focalloss
commonly used for multi-label text classifica- placesahigherweightoflosson“hard-to-classify”
tion (Bengio et al., 2013). Given a dataset instancespredictedwithlowprobabilityonground
8154

truth(Linetal.,2017). Forthemulti-labelclassifi- Table1: DatasetStatistics
cationtask,thefocallosscanbedefinedas:
Dataset Statistic
(cid:40)
−(1−pk)γlog(pk) ifyk =1
L FL = −(pk) γ lo i g(1−p i k) othe i rwise. (2) Reuters-21578
i i Numberofdocuments 10788
2.2 Class-balancedfocalloss(CB) Numberoflabels 90
Averagenumberoflabelsperinstance 1.24
By estimating the effective number of samples,
Averagenumberofinstancesperlabel 148.11
class-balancedfocalloss(Cuietal.,2019)further
PubMed
reweightsFLtocapturethediminishingmarginal
Numberofdocuments 224897
benefits of data, and therefore reduces redundant
Numberoflabels 18211
informationofheadclasses. Formulti-labeltasks,
Averagenumberoflabelsperinstance 12.30
eachlabelwithoverallfrequencyn hasitsbalanc-
i Averagenumberofinstancesperlabel 151.88
ingterm
1−β
r CB = 1−βni (3) whereq i k =σ(z i k−v i )forpositiveinstancesand
qk =σ(λ(zk −v ))fornegativeones. Thev can
whereβ ∈ [0,1)controlshowfasttheeffective i i i i
beestimatedbyminimizingthelossfunctionatthe
numbergrowsandthelossfunctionbecomes
beginningoftrainingwithascalefactorκandclass
(cid:40) priorp = n /N,sothat
−r (1−pk)γlog(pk) ifyk =1 i i
L CB = −r C C B B (pk i ) γ lo i g(1−p i k i ) othe i rwise. (4) ˆb =−log( 1 −1),v =−κ×ˆb (7)
i p i i
i
2.3 Distribution-balancedloss(DB)
Finally,DBintegratesrebalancedweightingand
Byintegratingrebalancedweightingandnegative-
NTRas
tolerant regularization (NTR), distribution-
balancedlossfirstreducesredundantinformation
(cid:40)
of label co-occurrence, which is critical in the L DB = − − r r ˆ ˆ DB ( 1 1 ( − qk) q γ i k l ) o γ g lo (1 g( − q i k q ) k) i o f th y e i k rw = is 1 e. (8)
multi-label scenario, and then explicitly assigns DBλ i i
lower weight on “easy-to-classify” negative
3 Experiments
instances(Wuetal.,2020).
First, to rebalance the weights, in the single- 3.1 Datasets
labelscenario,aninstancecanbeweightedbythe
Twomulti-labeltextclassificationdatasetsofdif-
resampling probability PC = 1 1 ; while in the
i Cni ferentsize,propertyanddomainareused(Table1).
multi-label scenario, if following the same strat-
Reuters-21578 dataset (Distribution 1.0) con-
egy,oneinstancewithmultiplelabelscanbeover-
tainsdocumentsthatappearedonReutersnewswire
sampled with a probability PI = 1 (cid:80) 1 .
C y
i
k=1 ni in1987andthatweremanuallyannotatedwith90
Therefore, the rebalanced weight can be normal-
labels(HayesandWeinstein,1990). Here,wefol-
izedwithr = PC/PI. Withasmoothingfunc-
DB i lowthetrain-testsplitusedby(YangandLiu,1999)
tion, rˆ = α + σ(β × (r − µ)), mapping
DB DB toobtain7769training(1000amongwhichforval-
r to[α,α+1],therebalanced-FL(R-FL)loss
DB idation)and3019testdocuments. Thelabelsare
functionisdefinedas:
equally split into head (30 with ≥ 35 instances),
medium(31withbetween8-35instances)andtail
(cid:40)
L R−FL = − − r r ˆ ˆ DB ( (p 1 k − ) γ p lo k i g )γ (1 lo − g(p p k i k ) ) i o f th y e i k rw = is 1 e. (5) (30with≤8instances)subsets.
DB i i PubMeddatasetcomesfromtheBioASQChal-
Then, NTR treats the positive and negative in- lenge (License Code: 8283NLM123) providing
stancesofthesamelabeldifferently. Ascalefactor PubMedarticleswithtitlesandabstracts,thathave
λ and an intrinsic class-specific bias v are intro- beenmanuallylabelledforMedicalSubjectHead-
i
ducedtolowerthethresholdfortailclassesandto ings (MeSH) (Tsatsaronis et al., 2015; Coordina-
avoidover-suppression. tors,2017). 224,897articlespublishedduring2020
and2021areused,amongwhich10,000areused
(cid:40) −(1−qk)γlog(qk) ifyk =1 forvalidationandtestingpurpose. The18,211la-
L NTR−FL = −1(qk) γ i log(1− i qk) othe i rwise. (6) belsaresplitby3-quantilesintohead(6018with
λ i i
8155

Table2:MicroandmacroF1scoresformulti-labelclassificationofReuters-21578(left)andPubMed(right)using
theSVMmodelordifferentlossfunctions. TheF1scoresarereportedforthetotalsetoflabelsaswellasforthe
head,mediumandtaillabelsets,withthenumberofinstancesgiveninparenthesis.Theexperimentsareperformed
withtheSVMone-vs-restmodel(SVM),thebinarycrossentropy(BCE),focalloss(FL),classbalancedfocalloss
(CB),rebalancedfocalloss(R-FL),negative-tolerantregularizationFL(NTR-FL),distributionbalancewithnoFL
(DB-0FL),classbalancedFLwithnegativeregularization(CB-NTR)anddistributionbalancedloss(DB).
Model/ Reuters Reuters Reuters Reuters PubMed PubMed PubMed PubMed
Loss Total Head(≥35) Med(8-35) Tail(≤8) Total Head(≥50) Med(15-50) Tail(≤15)
Function miF/maF miF/maF miF/maF miF/maF miF/maF miF/maF miF/maF miF/maF
SVM 87.60/51.63 89.87/78.47 66.92/61.00 22.54/13.83 58.54/13.31 60.77/34.33 19.78/5.62 6.94/0.67
BCE 89.14/47.32 91.75/82.81 66.28/57.26 0.00/0.00 26.17/0.02 27.61/0.06 0.00/0.00 0.00/0.00
FL 89.97/56.83 91.83/82.64 76.16/70.63 27.40/15.37 58.30/13.94 60.43/33.69 26.39/8.15 8.58/0.86
CB 89.23/52.96 91.56/80.44 71.64/66.61 23.08/9.93 58.57/13.67 60.75/33.40 24.50/7.39 9.92/1.01
R-FL 89.47/54.35 91.59/80.39 72.86/66.69 25.00/14.22 57.90/14.66 59.85/34.09 30.32/9.70 11.45/1.15
NTR-FL 90.70/60.70 92.37/82.65 79.35/75.34 39.51/22.33 60.92/16.99 63.15/38.85 33.14/11.39 15.86/1.82
DB-0FL 89.45/57.98 91.21/82.05 77.33/71.11 31.17/19.05 58.95/15.15 60.99/34.92 31.06/10.02 14.23/1.49
CB-NTR 90.74/63.31 92.46/83.28 78.42/72.98 46.32/32.31 61.07/18.40 63.02/39.95 37.18/13.43 24.15/2.97
DB 90.62/64.47 92.14/83.48 80.25/77.01 48.89/31.39 60.63/19.19 62.39/40.48 41.14/15.33 24.19/3.08
≥50instances),medium(5581withbetween15- 18,000labels(Table1)andtheimbalanceiseven
50instances)andtail(6612with≤15instances) morepronouncedforthislargedataset(Figurein
subsets. Appendix) and the difference between the total
micro-F1score(60)andthetotalmacro-F1score
3.2 ExperimentalSettings
(around 15) is very high. Overall, SVM under-
We compare the use of different loss functions, performs the proposed distribution balanced loss
and SVM one-vs-rest model as a classical multi- functionsinbothdatasets.
labelclassificationbaseline. Foreachdatasetand Experiments with Reuters-21578 dataset.
method,weevaluateitsbestmicro-F1andmacro- The loss functions FL, CB, R-FL and NTR-FL
F1 scores (Wu et al., 2019; Lipton et al., 2014) performsimilartoBCEinheadclasses,yetoutper-
for the whole label set (total) as well as different form BCE in medium and tail classes, indicating
subsetsoflabelfrequency(head/medium/tail). The theadvantageofhandlingimbalance. DBprovides
lossfunctionparameters,theclassificationmodels thebiggestimprovementintailclassassignment;
used,andtheimplementationdetailsareprovided the tail micro-F1 score gains 21.49 from FL and
inAppendixA. 25.81 from CB. It outperforms prior works that
also used this commonly used dataset, including
4 Results
approaches based on Binary Relevance, EncDec,
Asummaryoftheresultsofdifferentlossfunctions CNN, CNN-RNN, Optimal Completion Distilla-
arelistedinTable2. tionorattention-basedGNN,thatachievedmicro-
There are about 10,000 documents and 90 la- F1<89.9 (Nam et al., 2017; Pal et al., 2020; Tsai
belsintheReutersdataset,withanaverageof150 andLee,2020)
instances per label (Table 1). Figure 2 shows the Experiments with PubMed dataset. PubMed
long-tailed distribution where only a few labels isabiomedicaldomainspecific,largerdatasetwith
have a high number of articles and these head la- biggerclassimbalance. Forthisdataset,BCEdoes
belsalsohavehighco-occurrencewithotherlabels. notworkefficiently,thereforeweuseFLasastrong
Theimpactoftheskeweddistributioncanalsobe baseline. WithFL,themediumandtailmicro-F1
seen from the comparison between the micro-F1 scores are 26 and 9. All other loss functions out-
(around90fordifferentlossfunctions)andmacro- performFLinmediumandtailclasses,indicating
F1(around50-60)scores(Table2). Furthermore, theadvantageofbalancinglabeldistribution. DB
amonglossfunctions,BCEhasthelowestperfor- againhasthehighestperformanceforallclassesbut
mancefortheReutersdatasetwithtotalmacro-F1 the most significant improvement is achieved for
score of 47 and tail F1 scores of 0. The PubMed themedium(micro-F1:41)andtail(micro-F1:24)
dataset contains around 225,000 documents with classes.
8156

AblationStudy. Wefurtherinvestigatethecon- tion. Itdoesnotrequireadditionalinformationand
tribution of the three layers of DB by comparing canbeusedwithalltypesofneuralnetwork-based
DB results with R-FL, NTR-FL and DB without models. It may also be a powerful strategy for
the focal layer (DB-0FL). As shown in Table 2, other NLP tasks, such as part-of-speech tagging,
for both datasets, removing the NTR layer (R- namedentityrecognition,machinereadingcompre-
FL) or the focal layer (DB-0FL) reduces model hension,paraphraseidentificationandcoreference
performanceforallsubsets. Removingtherebal- resolution, all of which usually suffer from long-
| ancedweightinglayer(NTR-FL)yieldssimilarto- |           |     |     |            |     |         | taileddistribution. |     |     |     |     |     |
| ------------------------------------------- | --------- | --- | --- | ---------- | --- | ------- | ------------------- | --- | --- | --- | --- | --- |
| tal micro-F1                                | (Reuters: |     | 90, | PubMed:60) |     | but the |                     |     |     |     |     |     |
Acknowledgements
macro-F1aswellasmediumandtailF1scoresare
higherwithDB,showingthevalueofaddingthe
|                            |     |     |     |                   |     |     | We thank | Igor Kulev | for | the helpful | discussions, |     |
| -------------------------- | --- | --- | --- | ----------------- | --- | --- | -------- | ---------- | --- | ----------- | ------------ | --- |
| rebalancingweightinglayer. |     |     |     | Wealsotestthecon- |     |     |          |            |     |             |              |     |
andtheanonymousreviewersfortheirconstructive
tributionofNTRbyintegratingitwithCB,yielding
|         |               |     |        |      |     |          | suggestions. | TUBITAK-BIDEB2211-AScholar- |     |     |     |     |
| ------- | ------------- | --- | ------ | ---- | --- | -------- | ------------ | --------------------------- | --- | --- | --- | --- |
| a novel | loss function |     | CB-NTR | that | has | not been |              |                             |     |     |     |     |
shipProgram(toA.K.)andTUBA-GEBIPAward
| previouslyexplored. |     |     | Forbothdatasets, |     |     | CB-NTR |     |     |     |     |     |     |
| ------------------- | --- | --- | ---------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
oftheTurkishScienceAcademy(toA.O.)aregrate-
| has better | performance |     | than | CB for | all | class sets |     |     |     |     |     |     |
| ---------- | ----------- | --- | ---- | ------ | --- | ---------- | --- | --- | --- | --- | --- | --- |
fullyacknowledged.
| (Table2).                | TheonlydifferencebetweenCB-NTR |     |                         |     |              |     |            |     |     |     |     |     |
| ------------------------ | ------------------------------ | --- | ----------------------- | --- | ------------ | --- | ---------- | --- | --- | --- | --- | --- |
| andDBistheuseofCBweightr |                                |     |                         | CB  | insteadofthe |     |            |     |     |     |     |     |
| rebalancingweightrˆ      |                                |     | . DBhasverycloseperfor- |     |              |     |            |     |     |     |     |     |
|                          |                                | DB  |                         |     |              |     | References |     |     |     |     |     |
mancetooroutperformsCB-NTRinthemedium
|          |          |            |     |          |     |         | Rahul Agrawal, | Archit | Gupta, | Yashoteja   |          | Prabhu, and |
| -------- | -------- | ---------- | --- | -------- | --- | ------- | -------------- | ------ | ------ | ----------- | -------- | ----------- |
| and tail | classes, | suggesting |     | that the | rˆ  | weight, |                |        |        |             |          |             |
|          |          |            |     |          | DB  |         | Manik Varma.   |        | 2013.  | Multi-label | learning | with        |
which addresses the co-occurrence challenge, is millions of labels: Recommending advertiser bid
| useful. |     |     |     |     |     |     | phrases for | web | pages. | In Proceedings |     | of the 22nd |
| ------- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | -------------- | --- | ----------- |
Error Analysis. We perform an error analysis internationalconferenceonWorldWideWeb,pages
13–24.
andobservethatthemostcommonerrorsaredue
to incorrect classification to similar or linked la- Yoshua Bengio, Aaron Courville, and Pascal Vincent.
belsforalllossfunctions. Themostcommonthree 2013. Representation learning: A review and new
|     |     |     |     |     |     |     | perspectives. | IEEE | Transactions |     | on Pattern | Analy- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | ------------ | --- | ---------- | ------ |
pairsofclassesconfusedbyalllossfunctionsfor
sisandMachineIntelligence,35(8):1798–1828.
| theReutersdatasetare: |                    |     | platinumandgold,yenand |              |     |     |     |     |     |     |     |     |
| --------------------- | ------------------ | --- | ---------------------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| money-fx,             | platinumandcopper. |     |                        | ForthePubMed |     |     |     |     |     |     |     |     |
FranciscoCharte,AntonioJRivera,MaríaJdelJesus,
dataset, the most common errors are: Pandemics andFranciscoHerrera.2015. Addressingimbalance
|     |     |     |     |     |     |     | in multilabel | classification: |     | Measures |     | and random |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --------------- | --- | -------- | --- | ---------- |
andBetacoronavirus,PandemicsandSARS-CoV-2,
Neurocomputing,163:3–16.
resamplingalgorithms.
Pneumonia,ViralandBetacoronavirus,andBCE
hassignificantlymoreerrorsfortheseclassescom-
|     |     |     |     |     |     |     | Arman Cohan, | Sergey | Feldman, |     | Iz Beltagy, | Doug |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | -------- | --- | ----------- | ---- |
paredtotheotherinvestigatedlossfunctions. Downey, and Daniel Weld. 2020. SPECTER:
|     |     |     |     |     |     |     | Document-level    |     | representation |     | learning | using       |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | -------------- | --- | -------- | ----------- |
|     |     |     |     |     |     |     | citation-informed |     | transformers.  |     | In       | Proceedings |
5 Conclusion
|     |     |     |     |     |     |     | of the 58th       | Annual | Meeting      |     | of the | Association |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | ------ | ------------ | --- | ------ | ----------- |
|     |     |     |     |     |     |     | for Computational |        | Linguistics, |     | pages  | 2270–2282,  |
Weproposeandcomparetheapplicationofaseries
Online.AssociationforComputationalLinguistics.
ofbalancinglossfunctionstoaddresstheclassim-
|     |     |     |     |     |     |     | NCBI Resource | Coordinators. |     | 2017. | Database | re- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ------------- | --- | ----- | -------- | --- |
balanceprobleminmulti-labeltextclassification.
|            |           |      |          |          |         |        | sources      | of the  | National | Center    | for Biotechnology |            |
| ---------- | --------- | ---- | -------- | -------- | ------- | ------ | ------------ | ------- | -------- | --------- | ----------------- | ---------- |
| We first   | introduce | the  | loss     | function | DB      | to NLP |              |         |          |           |                   |            |
|            |           |      |          |          |         |        | Information. | Nucleic | Acids    | Research, |                   | 46(D1):D8– |
| and design | a novel   | loss | function |          | CB-NTR. | The    |              |         |          |           |                   |            |
D13.
| experiments | showthat |     | the DB | outperforms |     | other |                  |     |               |     |           |           |
| ----------- | -------- | --- | ------ | ----------- | --- | ----- | ---------------- | --- | ------------- | --- | --------- | --------- |
|             |          |     |        |             |     |       | Yin Cui, Menglin |     | Jia, Tsung-Yi |     | Lin, Yang | Song, and |
approachesbyconsideringlong-taileddistribution
|           |                |     |     |     |             |     | Serge Belongie.             |     | 2019. | Class-balanced |                | loss based |
| --------- | -------------- | --- | --- | --- | ----------- | --- | --------------------------- | --- | ----- | -------------- | -------------- | ---------- |
| and label | co-occurrence, |     | and | its | performance | is  |                             |     |       |                |                |            |
|           |                |     |     |     |             |     | oneffectivenumberofsamples. |     |       |                | In2019IEEE/CVF |            |
robusttodifferentdatasetssuchasReuters(90la-
ConferenceonComputerVisionandPatternRecog-
bels,generaldomain)andPubMed(18,211labels, nition(CVPR),pages9260–9269.
| biomedicaldomain). |     |     | Thisstudydemonstratesthat |     |     |     |               |          |     |        |        |          |
| ------------------ | --- | --- | ------------------------- | --- | --- | --- | ------------- | -------- | --- | ------ | ------ | -------- |
|                    |     |     |                           |     |     |     | Jacob Devlin, | Ming-Wei |     | Chang, | Kenton | Lee, and |
addressingchallengessuchasclassimbalanceand
|     |     |     |     |     |     |     | KristinaToutanova.2018. |     |     | Bert:Pre-trainingofdeep |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ----------------------- | --- | --- |
label co-occurrence through loss functions is an bidirectional transformers for language understand-
| effective | approach | for | multi-label |     | text classifica- |     |     |     |     |     |     |     |
| --------- | -------- | --- | ----------- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
ing. arXivpreprintarXiv:1810.04805.
8157

T.Durand,N.Mehrasa,andG.Mori.2019. Learninga FaustoMilletari,NassirNavab,andSeyed-AhmadAh-
deep convnet for multi-label classification with par- madi. 2016. V-net: Fully convolutional neural net-
tiallabels. In2019IEEE/CVFConferenceonCom- works for volumetric medical image segmentation.
puterVisionandPatternRecognition(CVPR),pages In 2016 Fourth International Conference on 3D Vi-
647–657,LosAlamitos,CA,USA.IEEEComputer sion(3DV),pages565–571.
Society.
|     |     |     |     |     |     | Jinseok Nam, | Eneldo | Loza | Mencía, | Hyunwoo |     | J Kim, |
| --- | --- | --- | --- | --- | --- | ------------ | ------ | ---- | ------- | ------- | --- | ------ |
AndrewEstabrooks,TaehoJo,andNathalieJapkowicz. and Johannes Fürnkranz. 2017. Maximizing sub-
2004. A multiple resampling method for learning setaccuracywithrecurrentneuralnetworksinmulti-
from imbalanced data sets. Computational intelli- labelclassification. InAdvancesinNeuralInforma-
gence,20(1):18–36. tion Processing Systems, volume 30. Curran Asso-
ciates,Inc.
| WeifengGe,SibeiYang,andYizhouYu.2018. |     |     |     |     | Multi- |     |     |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
evidencefilteringandfusionformulti-labelclassifi- Ankit Pal, Muru Selvakumar, and Malaikannan
cation, object detection and semantic segmentation Sankarasubbu.2020. Magnet: Multi-labeltextclas-
based on weakly supervised learning. In Proceed- sification using attention-based graph neural net-
ings of the IEEE Conference on Computer Vision work. InICAART(2),pages494–505.
andPatternRecognition(CVPR).
|                 |     |        |               |     |            | F. Pedregosa, | G.  | Varoquaux, |             | A. Gramfort, | V.               | Michel, |
| --------------- | --- | ------ | ------------- | --- | ---------- | ------------- | --- | ---------- | ----------- | ------------ | ---------------- | ------- |
|                 |     |        |               |     |            | B. Thirion,   | O.  | Grisel,    | M. Blondel, |              | P. Prettenhofer, |         |
| Philip J. Hayes | and | Steven | P. Weinstein. |     | 1990. Con- |               |     |            |             |              |                  |         |
strue/tis: A system for content-based indexing of a R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
databaseofnewsstories. InProceedingsoftheThe D.Cournapeau,M.Brucher,M.Perrot,andE.Duch-
SecondConferenceonInnovativeApplicationsofAr- esnay. 2011. Scikit-learn: Machine learning in
tificial Intelligence, IAAI ’90, page 49–64. AAAI Python. Journal of Machine Learning Research,
12:2825–2830.
Press.
|                |      |        |     |       |             | Yashoteja | Prabhu, | Anil | Kag, | Shrutendra | Harsola, |     |
| -------------- | ---- | ------ | --- | ----- | ----------- | --------- | ------- | ---- | ---- | ---------- | -------- | --- |
| Gakuto Kurata, | Bing | Xiang, | and | Bowen | Zhou. 2016. |           |         |      |      |            |          |     |
Improved neural network-based multi-label classifi- Rahul Agrawal, and Manik Varma. 2018. Parabel:
cation with better initialization leveraging label co- Partitionedlabeltreesforextremeclassificationwith
occurrence. InProceedingsofthe2016Conference application to dynamic search advertising. In Pro-
of the North American Chapter of the Association ceedings of the 2018 World Wide Web Conference,
pages993–1002.
| for Computational |     | Linguistics: |     | Human | Language |     |     |     |     |     |     |     |
| ----------------- | --- | ------------ | --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- |
Technologies,pages521–526.
|     |     |     |     |     |     | Che-Ping | Tsai | and Hung-yi |     | Lee. 2020. | Order-free |     |
| --- | --- | --- | --- | --- | --- | -------- | ---- | ----------- | --- | ---------- | ---------- | --- |
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, learningalleviatingexposurebiasinmulti-labelclas-
Donghyeon Kim, Sunkyu Kim, Chan Ho So, sification. In The Thirty-Fourth AAAI Conference
and Jaewoo Kang. 2019. BioBERT: a pre-trained on Artificial Intelligence, AAAI 2020, The Thirty-
|            |          |     |                |     |           | Second | Innovative | Applications |     | of Artificial |     | Intelli- |
| ---------- | -------- | --- | -------------- | --- | --------- | ------ | ---------- | ------------ | --- | ------------- | --- | -------- |
| biomedical | language |     | representation |     | model for |        |            |              |     |               |     |          |
genceConference,IAAI2020,TheTenthAAAISym-
| biomedicaltextmining. |     |     | Bioinformatics. |     |     |        |                |     |          |     |            |        |
| --------------------- | --- | --- | --------------- | --- | --- | ------ | -------------- | --- | -------- | --- | ---------- | ------ |
|                       |     |     |                 |     |     | posium | on Educational |     | Advances | in  | Artificial | Intel- |
Jianqiang Li, Guanghui Fu, Yueda Chen, Pengzhi Li, ligence, EAAI 2020, New York, NY, USA, February
Bo Liu, Yan Pei, and Hui Feng. 2020a. A multi- 7-12,2020,pages6038–6045.AAAIPress.
| label classification |     | model | for | full slice | brain com- |     |     |     |     |     |     |     |
| -------------------- | --- | ----- | --- | ---------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
puterised tomography image. BMC Bioinformatics, George Tsatsaronis, Georgios Balikas, Prodromos
|     |     |     |     |     |     | Malakasiotis, |     | Ioannis | Partalas, | Matthias | Zschunke, |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ------- | --------- | -------- | --------- | --- |
21(6):200.
|     |     |     |     |     |     | Michael | R Alvers, |     | Dirk | Weissenborn, | Anastasia |     |
| --- | --- | --- | --- | --- | --- | ------- | --------- | --- | ---- | ------------ | --------- | --- |
Xiaoya Li, Xiaofei Sun, Yuxian Meng, Junjun Liang, Krithara, Sergios Petridis, Dimitris Polychronopou-
Fei Wu, and Jiwei Li. 2020b. Dice loss for data- los, Yannis Almirantis, John Pavlopoulos, Nico-
imbalanced NLP tasks. In Proceedings of the 58th las Baskiotis, Patrick Gallinari, Thierry Artieres,
Annual Meeting of the Association for Computa- Axel Ngonga, Norman Heino, Eric Gaussier, Lil-
tional Linguistics, pages 465–476, Online. Associ- iana Barrio-Alvers, Michael Schroeder, Ion An-
|     |     |     |     |     |     | droutsopoulos, |     | and Georgios |     | Paliouras. | 2015. | An  |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | --- | ---------- | ----- | --- |
ationforComputationalLinguistics.
|     |     |     |     |     |     | overview | of  | the bioasq | large-scale |     | biomedical | se- |
| --- | --- | --- | --- | --- | --- | -------- | --- | ---------- | ----------- | --- | ---------- | --- |
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming mantic indexing and question answering competi-
He,andPiotrDollár.2017. Focallossfordenseob- tion. BMCBioinformatics,16:138.
| ject detection. |     | In 2017 | IEEE | International | Confer- |     |     |     |     |     |     |     |
| --------------- | --- | ------- | ---- | ------------- | ------- | --- | --- | --- | --- | --- | --- | --- |
enceonComputerVision(ICCV),pages2999–3007, Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
|     |     |     |     |     |     | Chaumond, | ClementDelangue, |     |     | AnthonyMoi, |     | Pier- |
| --- | --- | --- | --- | --- | --- | --------- | ---------------- | --- | --- | ----------- | --- | ----- |
LosAlamitos,CA,USA.IEEEComputerSociety.
|     |     |     |     |     |     | ric Cistac, | Tim | Rault, | Rémi | Louf, Morgan |     | Funtow- |
| --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | ---- | ------------ | --- | ------- |
Zachary C. Lipton, Charles Elkan, and Balakrishnan icz, Joe Davison, Sam Shleifer, Patrick von Platen,
Naryanaswamy.2014. Optimalthresholdingofclas- Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu,
sifiers to maximize f1 measure. In Machine Learn- Teven Le Scao, Sylvain Gugger, Mariama Drame,
ing and Knowledge Discovery in Databases, pages Quentin Lhoest, and Alexander M. Rush. 2020.
225–239, Berlin, Heidelberg. Springer Berlin Hei- Transformers: State-of-the-artnaturallanguagepro-
| delberg. |     |     |     |     |     | cessing. | InProceedingsofthe2020Conferenceon |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | -------- | ---------------------------------- | --- | --- | --- | --- | --- |
8158

EmpiricalMethodsinNaturalLanguageProcessing:
SystemDemonstrations,pages38–45,Online.Asso-
ciationforComputationalLinguistics.
Jiawei Wu, Wenhan Xiong, and William Yang Wang.
2019. Learning to learn and predict: A meta-
learning approach for multi-label classification. In
Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 4354–
4364,HongKong,China.AssociationforComputa-
tionalLinguistics.
Tong Wu, Qingqiu Huang, Ziwei Liu, Yu Wang, and
Dahua Lin. 2020. Distribution-balanced loss for
multi-label classification in long-tailed datasets. In
Computer Vision – ECCV 2020, pages 162–178,
Cham.SpringerInternationalPublishing.
Wenshuo Yang, Jiyi Li, Fumiyo Fukumoto, and Yan-
ming Ye. 2020. HSCNN: A hybrid-Siamese con-
volutionalneuralnetworkforextremelyimbalanced
multi-labeltextclassification. InProceedingsofthe
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6716–6722,
Online.AssociationforComputationalLinguistics.
Yiming Yang and Xin Liu. 1999. A re-examination
of text categorization methods. In Proceedings of
the 22nd Annual International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval, SIGIR ’99, page 42–49, New York, NY,
USA.AssociationforComputingMachinery.
8159

A Appendix
A.1 ExperimentalSettings
| Evaluationmetrics. |     | Foreachdatasetandmethod, |     |     |     |     |     |     |     |     |
| ------------------ | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
weselectthethresholdwiththebestmicro-F1score
onthevalidationsetasourfinalmodelandevaluate
itsperformanceonthetestsetwithmicro-F1and
macro-F1scores.
| Loss function                       |                                 | parameters.             | We                 | compare         | the   |     |     |     |     |     |
| ----------------------------------- | ------------------------------- | ----------------------- | ------------------ | --------------- | ----- | --- | --- | --- | --- | --- |
| performance                         | of                              | DB with                 | different          | loss functions, |       |     |     |     |     |     |
| whereBCEoritsmodificationsareused.  |                                 |                         |                    | Themeth-        |       |     |     |     |     |     |
| odsinclude:                         | (1)BCEwithallinstancesandlabels |                         |                    |                 |       |     |     |     |     |     |
| ofthesameweight.                    |                                 | (2)FL(Linetal.,2017):   |                    |                 | we    |     |     |     |     |     |
| useγ=2.                             |                                 |                         |                    | weuseβ          |       |     |     |     |     |     |
|                                     | (3)CB(Cuietal.,2019):           |                         |                    |                 | =0.9. |     |     |     |     |     |
| (4)R-FL(Wuetal.,2020):              |                                 |                         | weuseα=0.1andβ=10, |                 |       |     |     |     |     |     |
| µ=0.9(Reuters-21578)or0.05(PubMed). |                                 |                         |                    | (5)NTR-         |       |     |     |     |     |     |
| FL(Wuetal.,2020):                   |                                 | weuseκ=0.05andλ=2.      |                    |                 | (6)   |     |     |     |     |     |
| DB(Wuetal.,2020):                   |                                 | weusesameparameterswith |                    |                 |       |     |     |     |     |     |
R-FLandNTR-FLwhenapplicable.
| ImplementationDetails. |     |     | WeusetheBertForSe- |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
quenceClassificationbackboneintransformersli-
brary(Wolfetal.,2020)withthebert-base-cased
pretrainedmodel(Devlinetal.,2018)forReuters-
21578datasetandthebiobert-base-cased-v1.1pre-
trainedmodel(Leeetal.,2019)forPubMeddataset.
|     |     |     |     |     |     | Figure 3: | The long-tailed | distribution | and label | co- |
| --- | --- | --- | --- | --- | --- | --------- | --------------- | ------------ | --------- | --- |
bert-base-cased and biobert-base-cased-v1.1 are occurrenceforthePubMeddataset.They-axisofdistri-
base BERT models with 110 million parameters. butioncurveislog-scale, andtheco-occurencematrix
iscolorcodedbasedonthequadroot(forbettervisual-
| The training | data | are truncated | with | a maximal |     |     |     |     |     |     |
| ------------ | ---- | ------------- | ---- | --------- | --- | --- | --- | --- | --- | --- |
ization)ofconditionalprobabilityp(i|j)ofclassinthe
lengthof512andgroupedwithabatchsizeof32.
ithcolumnonclassinthejthrow.
| We use AdamW   |     | with a    | weight decay | of       | 0.01 as |     |     |     |     |     |
| -------------- | --- | --------- | ------------ | -------- | ------- | --- | --- | --- | --- | --- |
| the optimizer, | and | determine | the          | learning | rate by |     |     |     |     |     |
hyperparametersearch. Theexperimentsareimple- functions in micro-F1 of the multi-label instance
mentedinPyTorch. ForReuters-21578datasetwe group and macro-F1 of both groups. There are <
use one-GPU (V100) experiments which takes 5 0.1% instances of PubMed dataset with a single
| minutesforoneepoch. |     | ForPubMeddataset,weuse |     |     |     |           |                     |      |             |     |
| ------------------- | --- | ---------------------- | --- | --- | --- | --------- | ------------------- | ---- | ----------- | --- |
|                     |     |                        |     |     |     | label, so | we divide instances | into | 3-quantiles | by  |
one-GPU(A100)experimentswhichtakes1hour theirnumberoflabels. Ineachquantile,thenovel
for one epoch. For the SVM one-vs-rest model, NTR-FL,CB-NTRandDBoutperformtherestof
weusescikit-learnlibrary(Pedregosaetal.,2011) themodelsinallmetrics.
| withTF-IDFfeatures. |     | Withhyperparametersearch, |     |     |     |     |     |     |     |     |
| ------------------- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
weapplythelinearkernelandhyper-planeshifting
optimizedoneachvalidationset.
A.2 AdditionalEffectivenessCheck
Wefurtherinvestigatetheeffectivenessoflossfunc-
tionsagainstthenumberoflabelsperinstance(Ta-
| ble 3 in Appendix). |                | For  | the Reuters | dataset, | we  |     |     |     |     |     |
| ------------------- | -------------- | ---- | ----------- | -------- | --- | --- | --- | --- | --- | --- |
| split the           | test instances | into | two groups, | 2583     | in- |     |     |     |     |     |
stanceswithonlyonelabeland436instanceswith
| multiplelabels. | Onsingle-labelinstances,allfunc- |     |     |     |     |     |     |     |     |     |
| --------------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionsfromBCEtoDB,havesimilarperformance;
whileonmulti-labelinstances,theperformanceof
BCEdropsmorethanDB.DBoutperformsother
8160

Table 3: Micro and macro F1 scores for multi-label classification of Reuters-21578 (left) and PubMed (right)
using different loss functions. The F1 scores are reported for the total set of labels as well as for groups split
bythenumberoflabelsperinstance. Theexperimentsareperformedwiththebinarycrossentropy(BCE),focal
loss(FL),classbalancedfocalloss(CB),rebalancedfocalloss(R-FL),negative-tolerantregularizationFL(NTR-
FL), distribution balance with no FL (DB-0FL), class balanced FL with negative regularization (CB-NTR) and
distributionbalancedloss(DB).
Reuters Reuters Reuters PubMed PubMed PubMed PubMed
Loss
Total Single-label Multi-label Total ≤9labels 10-14labels ≥15labels
Function
miF/maF miF/maF miF/maF miF/maF miF/maF miF/maF miF/maF
BCE 89.14/47.32 94.11/41.44 76.26/33.11 26.17/0.02 16.48/0.01 27.36/0.02 30.36/0.03
FL 89.97/56.83 94.81/50.33 77.54/40.07 58.30/13.94 53.72/7.44 59.02/10.27 59.72/8.63
CB 89.23/52.96 94.10/44.72 77.27/38.80 58.57/13.67 54.41/7.40 59.21/10.11 59.82/8.51
R-FL 89.47/54.35 95.21/47.45 74.29/38.79 57.90/14.66 53.08/7.67 58.60/10.50 59.45/8.81
NTR-FL 90.70/60.70 95.42/51.33 78.85/44.37 60.92/16.99 58.51/9.07 61.86/12.31 61.12/10.20
DB-0FL 89.45/57.98 94.48/51.80 76.63/42.26 58.95/15.15 55.14/8.11 59.84/10.90 59.85/8.94
CB-NTR 90.74/63.31 95.17/51.08 79.56/49.94 61.07/18.40 58.29/9.67 61.72/12.97 61.72/10.77
DB 90.62/64.47 94.49/54.31 81.17/50.12 60.63/19.19 57.81/9.76 61.53/13.49 61.08/11.23
8161
