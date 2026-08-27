Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
Taming Pretrained Transformers
for Extreme Multi-label Text Classification
Wei-ChengChang Hsiang-FuYu KaiZhong
CarnegieMellonUniversity Amazon Amazon
YimingYang InderjitS.Dhillon
CarnegieMellonUniversity Amazon&UTAustin
ABSTRACT 1 INTRODUCTION
Weconsidertheextrememulti-labeltextclassification(XMC)prob- We are interested in the Extreme multi-label text classification
lem:givenaninputtext,returnthemostrelevantlabelsfromalarge (XMC)problem:givenaninputtextinstance,returnthemostrele-
labelcollection.Forexample,theinputtextcouldbeaproductde- vantlabelsfromanenormouslabelcollection,wherethenumber
scriptiononAmazon.comandthelabelscouldbeproductcategories. oflabelscouldbeinthemillionsormore.OnecanviewtheXMC
XMCisanimportantyetchallengingproblemintheNLPcommu- problemaslearningascorefunctionf :X×Y →R,thatmapsan
nity.Recently,deeppretrainedtransformermodelshaveachieved (instance,label)pair(x,y)toascoref(x,y).Thefunctionf should
state-of-the-artperformanceonmanyNLPtasksincludingsen- beoptimizedsuchthathighlyrelevant(x,y)pairshavehighscores,
tenceclassification,albeitwithsmalllabelsets.However,naively whereastheirrelevantpairshavelowscores.Manyreal-worldap-
applyingdeeptransformermodelstotheXMCproblemleadsto plicationsareinthisform.Forexample,inE-commercedynamic
sub-optimalperformanceduetothelargeoutputspaceandthelabel search advertising, x represents an item and y represents a bid
sparsityissue.Inthispaper,weproposeX-Transformer,thefirst queryonthemarket[20,21].Inopen-domainquestionanswering,
scalableapproachtofine-tuningdeeptransformermodelsforthe xrepresentsaquestionandyrepresentsanevidencepassagecon-
XMCproblem.Theproposedmethodachievesnewstate-of-the-art tainingtheanswer[4,11].InthePASCALLarge-ScaleHierarchical
resultsonfourXMCbenchmarkdatasets.Inparticular,onaWiki TextClassification(LSHTC)challenge,xrepresentsanarticleandy
datasetwitharound0.5millionlabels,theprec@1ofX-Transformer representsacategoryoftheWikipediahierarchicaltaxonomy[17].
is77.28%,asubstantialimprovementoverstate-of-the-artXMCap- XMCisessentiallyatextclassificationproblemonanindustrial
proachesParabel(linear)andAttentionXML(neural),whichachieve scale,whichisoneofthemostimportantandfundamentaltopicsin
68.70%and76.95%precision@1,respectively.WefurtherapplyX- machinelearningandnaturallanguageprocessing(NLP)communi-
Transformertoaproduct2querydatasetfromAmazonandgained ties.Recently,deeppretrainedTransformers,e.g.,BERT[5],along
10.7%relativeimprovementonprec@1overParabel. withitsmanysuccessorssuchasXLNet[30]andRoBERTa[13],
haveledtostate-of-the-artperformanceonmanytasks,suchas
questionanswering,part-of-speechtagging,informationretrieval,
CCSCONCEPTS
andsentenceclassificationwithveryfewlabels.Deeppretrained
•Computingmethodologies→Machinelearning;Naturallan-
Transformermodelsinducepowerfultoken-levelandsentence-level
guageprocessing;•Informationsystems→Informationretrieval.
embeddingsthatcanberapidlyfine-tunedonmanydownstream
NLPproblemsbyaddingatask-specificlightweightlinearlayeron
KEYWORDS topofthetransformermodels.
However,howtosuccessfullyapplyTransformermodelstoXMC
Transformermodels,eXtremeMulti-labeltextclassification
problemsremainsanopenchallenge,primarilyduetotheextremely
largeoutputspaceandseverelabelsparsityissues.Asaconcrete
ACMReferenceFormat:
example,Table1comparesthemodelsize(intermsofthenumber
Wei-ChengChang,Hsiang-FuYu,KaiZhong,YimingYang,andInderjit
ofmodelparameters)andGPUmemoryusage,whenapplyinga24-
S.Dhillon.2020.TamingPretrainedTransformersforExtremeMulti-label
TextClassification.InProceedingsofthe26thACMSIGKDDConferenceon layerXLNetmodeltoabinaryclassificationproblem(e.g.,theMNLI
datasetofGLUE[27])versusitsapplicationtoanXMCproblemwith
KnowledgeDiscoveryandDataMining(KDD’20),August23–27,2020,Virtual
Event,CA,USA.ACM,NewYork,NY,USA,9pages.https://doi.org/10.1145/ 1millionlabels.NotethattheclassifierfortheMNLIproblemand
3394486.3403368 XMCproblemhasamodelsizeof2Kand1025M,respectively.This
meansthatthelatterisamuchharderproblemthantheformerfrom
themodeloptimizationpointofview.Additionally,inattempting
tosolvetheXMCproblem,werunoutofGPUmemoryevenfora
Permissiontomakedigitalorhardcopiesofpartorallofthisworkforpersonalor singleexamplemini-batchupdate.Table1givesthedetailsofthe
classroomuseisgrantedwithoutfeeprovidedthatcopiesarenotmadeordistributed
GPUmemoryusageinthetrainingstagesofoneforwardpass,one
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation
onthefirstpage.Copyrightsforthird-partycomponentsofthisworkmustbehonored. backwardpassandoneoptimizationstep,respectively.
Forallotheruses,contacttheowner/author(s). Inadditiontothecomputationalchallenges,thelargeoutput
KDD’20,August23–27,2020,VirtualEvent,CA,USA
spaceinXMCisexacerbatedbyaseverelabelsparsityissue.The
©2020Copyrightheldbytheowner/author(s).
ACMISBN978-1-4503-7998-4/20/08. leftpartofFigure1illustratesthe“long-tailed”labeldistribution
https://doi.org/10.1145/3394486.3403368
3163

Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
XLNet-largemodel(#params) (batchsize,sequencelength)=(1,128)
problem encoder classifier total loadmodel +forward +backward +optimizerstep
GLUE(MNLI) 361M 2K 361M 2169MB 2609MB 3809MB 6571MB
XMC(1M) 361M 1,025M 1,386M 6077MB 6537MB OOM OOM
Table1:Ontheleftofarethemodelsizes(numbersofparameters)whenapplyingtheXLNet-largemodeltotheMNLIproblem
vs.theXMC(1M)problem;ontherightistheGPUmemoryusage(inmegabytes)insolvingthetwoproblems,respectively.
TheresultswereobtainedonarecentNvidia2080TiGPUwith12GBmemory.OOMstandsforout-of-memory.
intheWiki-500Kdataset[25].Only2%ofthelabelshavemore Matchingcomponentfine-tunesaTransformermodelforeachof
than100traininginstances,whiletheremaining98%arelong-tail theSLI-inducedXMCsub-problems,resultinginabettermapping
labelswithmuchfewertraininginstances.Howtosuccessfully fromtheinputtexttothesetoflabelclusters.Finally,theEnsemble
fine-tuneTransformermodelswithsuchsparselylabeleddataisa Rankingcomponentistrainedconditionallyontheinstance-to-
toughquestionthathasnotbeenwell-studiedsofar,tothebestof clusterassignmentandneuralembeddingfromtheTransformer,
ourknowledge. andisusedtoassemblescoresderivedfromvariousSLI-induced
sub-problemsforfurtherperformanceimprovement.
Inourexperiments,theproposedX-Transformerachievesnew
state-of-the-artresultsonfourXMCbenchmarksandleadstoim-
provementontworeal-wouldXMCapplications.OnaWikidataset
withahalfmillionlabels,theprecision@1ofX-Transformerreaches
77.28%,asubstantialimprovementoverthewell-establishedhierar-
chicallabeltreeapproachParabel[20](i.e.,68.70%)andthecompet-
ingdeeplearningmethodAttentionXML[32](i.e.,76.95%).Further-
more,X-Transformeralsodemonstratesgreatimpactonthescalabil-
ityofdeepTransformermodelsinreal-worldlargeapplications.In
ourapplicationofX-TransformertoAmazonProduct2Queryprob-
Figure1:Ontheleft,Wiki-500Kshowsalong-taildistribution
lemthatcanbeformulatedasXMC,X-Transformersignificantly
oflabels.Only2.1%ofthelabelshavemorethan100training
outperformsParabeltoo.Thedataset,experimentcode,modelsare
available:https://github.com/OctoberChang/X-Transformer.
instances,asindicatedbythecyanblueregime.Ontheright
istheclustersdistributionafteroursemanticlabelindexing
basedondifferentlabelrepresentations;99.4%oftheclusters 2 RELATEDWORKANDBACKGROUND
havemorethan100traininginstances,whichmitigatesthe 2.1 ExtremeMulti-labelClassification
datasparsityissueforfine-tuningofTransformermodels.
SparseLinearModels. Toovercomecomputationalissues,most
Instead of fine-tuning deep Transformer models and dealing existing XMC algorithms use sparse TF-IDF features (or slight
withthebottleneckclassifierlayer,analternativeistouseamore variants), and leverage different partitioning techniques on the
economicaltransferlearningparadigmasstudiedinthecontext labelspacetoreducecomplexity.Forexample,sparselinearone-vs-
ofword2vec[15],ELMo[19],andGPT[22].Forinstance,ELMo all(OVA)methodssuchasDiSMEC[1],ProXML[2]andPPDSparse
usesa(bi-directionalLSTM)modelpretrainedonlargeunlabeled [31]exploreparallelismtospeedupthealgorithmandreducethe
textdatatoobtaincontexualizedwordembeddings.Whenapplying modelsizebytruncatingmodelweightstoencouragesparsity.OVA
ELMoonadownstreamtask,thesewordembeddingscanbeusedas approachesarealsowidelyusedasbuildingblocksformanyother
inputwithoutadaptation.ThisisequivalenttofreezingtheELMo approaches,forexample,inParabel[20]andSLICE[7],linearOVA
encoder,andfine-tuningthedownstreamtask-specificmodelon classifierswithsmalleroutputdomainsareused.
topofELMo,whichismuchmoreefficientintermsofmemoryas Theefficiencyandscalabilityofsparselinearmodelscanbefur-
wellascomputation.However,suchabenefitcomesatthepriceof therimprovedbyincorporatingdifferentpartitioningtechniques
limitingthemodelcapacityfromadaptingtheencoder,aswewill onthelabelspaces.Forinstance,Parabel[20]partitionsthelabels
seeintheexperimentalresultsinSection4. throughabalanced2-meanslabeltreeusinglabelfeaturescon-
Inthispaper,weproposeX-Transformer,anewapproachthat structedfromtheinstances.Recently,severalapproachesarepro-
overcomestheaforementionedissues,withsuccessfulfine-tuning posedtoimproveParabel.Bonsai[9]relaxestwomainconstraints
ofdeepTransformermodelsfortheXMCproblem.X-Transformer inParabel:1)allowingmulti-wayinsteadofbinarypartitionings
consistsofaSemanticLabelIndexingcomponent,aDeepNeural ofthelabelsetateachintermediatenode,and2)removingstrict
Matchingcomponent,andanEnsembleRankingcomponent.First, balancingconstraintsonthepartitions.SLICE[7]considersbuild-
SemanticlabelIndexing(SLI)decomposestheoriginalintractable inganapproximatenearestneighbor(ANN)graphasanindexing
XMCproblemintoasetoffeasiblesub-problemsofmuchsmaller structureoverthelabels.Foragiveninstance,therelevantlabels
outputspacevialabelclustering,whichmitigatesthelabelsparsity canbefoundquicklyfromthenearestneighborsoftheinstance
issueasshownintherightpartofFigure1.Second,theDeepNeural viatheANNgraph.
3164

Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
DeepLearningApproaches. InsteadofusinghandcraftedTF- 2.3 AmazonApplications
IDFfeatureswhicharehardtooptimizefordifferentdownstream ManychallengingproblemsatAmazonamounttofindingrelevant
XMCproblems,deeplearningapproachesemployvariousneural resultsfromanenormousoutputspaceofpotentialcandidates:for
networkarchitecturestoextractsemanticembeddingsofthein- example,suggestingkeywordstoadvertisersstartingnewcam-
puttext.XML-CNN[12]employsone-dimensionalConvolutional paignsonAmazon,predictingnextqueriesacustomerwilltype
neuralnetworksalongbothsequencelengthandwordembedding basedonthepreviousquerieshe/shetyped.Herewediscusskey-
dimensionforrepresentingtextinput.Asafollow-up,SLICEcon- wordrecommendationsystemforAmazonSponsoredProducts,
sidersdenseembeddingfromthesupervisedpre-trainedXML-CNN as illustrations in Fig.2, and how it can be formulated as XMC
modelsastheinputtoitshierarchicallinearmodels.Morerecently, problems.
AttentionXML[32]usesBiLSTMsandlabel-awareattentionasthe
scoringfunction,andperformswarm-uptrainingofthemodels Keywordrecommendationsystem. KeywordRecommenda-
withhierarchicallabeltrees.Inaddition,AttentionXMLconsider tionSystemsprovidekeywordsuggestionsforadvertiserstocreate
variousnegativesamplingstrategiesonthelabelspacetoavoid campaigns.Inordertomaximizethereturnofinvestmentforthe
back-propagatingtheentirebottleneckclassifierlayer. advertisers,thesuggestedkeywordsshouldbehighlyrelevantto
theirproductssothatthesuggestionscanleadtoconversion.An
XMCmodel,whentrainedonanproduct-to-querydatasetsuchas
product-querycustomerpurchaserecords,cansuggestqueriesthat
2.2 TransferLearningApproachesinNLP arerelevanttoanygivenproductbyutilizingproductinformation,
Recently,theNLPcommunityhaswitnessedadramaticparadigm liketitle,description,brand,etc.
shifttowardsthe“pre-trainingthenfine-tuning”framework.One
ofthepioneeringworksisBERT[5],whosepre-trainingobjectives
aremaskedtokenpredictionandnextsentencepredictiontasks.
After pre-training on large-scale unsupervised corpora such as
WikipediaandBookCorpus,theTransformermodeldemonstrates
vastimprovementoverexistingstate-of-the-artwhenfine-tuned
on many NLP tasks such as the GLUE benchmark [27], named
entityrecognition,andquestionanswering.Moreadvancedvari-
antsofthepre-trainedTransformermodelsincludeXLNet[30]and
RoBERTa[13].XLNetconsiderspermutationlanguagemodelingas
thepre-trainingobjectiveandtwo-streamself-attentionfortarget-
awaretokenprediction.Itisworthnotingthatthecontextualized
tokenembeddingsextractedfromXLNetalsodemonstratecompet-
itiveperformancewhenfedintoatask-specificdownstreammodel Figure2:keywordrecommendationsystem
onlarge-scaleretrievalproblems.RoBERTaimprovesuponBERT
byusingmorerobustoptimizationwithlarge-batchsizeupdate,
andpre-trainingthemodelforlongertillittrulyconverges. 3 PROPOSEDMETHOD:X-TRANSFORMER
However,transferringthesuccessofthesepre-trainedTrans-
3.1 ProblemFormulation
formermodelsontheGLUEtextclassificationtotheXMCproblem
isnon-trivial,asweillustratedinTable1.Beforetheemergenceof Motivations. GivenatrainingsetD = {(xi,yi)|xi ∈ X,yi ∈
BERT-typeend-to-endfine-tuning,thecanonicalwayoftransfer {0,1}L,i = 1,...,N}, extreme multi-label classification aims to
learninginNLPperhapscomesfromthewell-knownWord2Vec[15] learnascoringfunctionf thatmapsaninput(orinstance)xi anda
orGloVe[18]papers.Word2vecisashallowtwo-layerneuralnet- labell toascoref(xi,l) ∈R.Thefunctionf shouldbeoptimized
workthatistrainedtoreconstructthelinguisticcontextofwords. suchthatthescoreishighwheny = 1(i.e.,labell isrelevant
il
GLoVeconsidersamatrixfactorizationobjectivetoreconstructthe toinstancexi )andthescoreislowwheny
il
= 0.Asimpleone-
globalword-to-wordco-occurrenceinthecorpus.Acriticaldown- versus-allapproachrealizesthescoringfunctionf as
sideofWord2vecandGloVeisthatthepre-trainedwordembed- f(x,l)=wTϕ(x)
dingsarenotcontextualizeddependingonthelocalsurrounding l
word.ELMo[19]andGPT2[22]insteadpresentcontextualized whereϕ(x) represents an encoding and W = [w1,...,wL ]T ∈
wordembeddingsbyusinglargeBiLSTMorTransformermodels. RL×d istheclassifierbottlenecklayer.Forconvenience,wefurther
Afterthemodelsarepre-trained,transferlearningcanbeeasily definethetop-bpredictionoperatoras
carriedoutbyfeedingtheseextractedwordembeddingsasinput
(cid:18)(cid:102) (cid:103)(cid:19)
f (x)=Top-b f(x,1),...,f(x,L) ∈{1,...,L},
tothedownstreamtask-specificmodels.Thisismoreefficientcom- b
paredtotheBERT-likeend-to-endadditionalfine-tuningofthe wheref (x)isanindexsetcontainingthetop-bpredictedlabels.
b
encoder,butcomesattheexpenseoflosingmodelexpressiveness. AswepointedoutinTable1,itisnotonlyverydifficulttofine-tune
Intheexperimentalresultssection,weshowthatusingfixedword theTransformerencodersϕT(x;θ)togetherwiththeintractable
embeddingsfromuniversalpre-trainedmodelssuchasBERTisnot classifierlayerW,butalsoextremelyslowtocomputethetop-K
powerfulenoughforXMCproblems. predictedlabelsefficiently.
3165

Applied Data Science Track Paper   KDD '20, August 23–27, 2020, Virtual Event, USA
Figure3:TheproposedX-Transformerframework.First,SemanticLabelIndexingreducesthelargeoutputspace.Transform-
ers are then fine-tuned on the XMC sub-problem that maps instances to label clusters. Finally, linear rankers are trained
conditionallyontheclustersandTransformer’soutputinordertore-rankthelabelswithinthepredictedclusters.
High-levelSketch. Tothisend,weproposeX-Transformeras Givenalabelrepresentation,weclustertheLlabelshierarchically
apracticalsolutiontofine-tunedeepTransformermodelsonXMC toformahierarchicallabeltreewithKleafnodes[7,9,20,32].For
problems.Figure3summarizesourproposedframework. simplicity,weconsiderbinarybalancedhierarchicaltrees[14,20]
Inanutshell,X-TransformerdecomposestheintractableXMC asthedefaultsetting.Duetothelackofadirectandinformative
problem to a feasible sub-problem with a smaller output space, representationofthelabels,theindexingsystemforXMCmaybe
whichisinducedfromsemanticlabelindexing,whichclustersthe noisy.Fortunately,theinstancesinXMCaretypicallyveryinforma-
labels.Werefertothissub-problemastheneuralmatcherofthe tive.Therefore,wecanutilizetherichinformationoftheinstances
followingform: tobuildastrongmatchingsystemaswellasastrongrankerto
compensatefortheindexingsystem.
|     | д(x,k)=wT ϕT(x), | k =1,...,K |     | (1) |     |     |     |
| --- | ---------------- | ---------- | --- | --- | --- | --- | --- |
k
Giventextinformationabout
whereK isthenumberofclusterswhichissignificantlysmaller Labelembeddingvialabeltext.
labels,suchasashortdescriptionofcategoriesintheWikipedia
thantheoriginalintractableXMCproblemofsizeO(L).Finally,
currently uses a linear ranker that conditionally datasetorsearchqueriesontheAmazonshoppingwebsite,wecan
X-Transformer
dependsontheembeddingoftransformermodelsanditstop-b usethisshorttexttorepresentthelabels.Inthiswork,weusea
pretrainedXLNet[19]torepresentthewordsinthelabel.Thelabel
| predictedclustersд | (x).     |          |     |                                                     |     |     |     |
| ------------------ | -------- | -------- | --- | --------------------------------------------------- | --- | --- | --- |
|                    | b        |          |     | embeddingisthemeanpoolingofallXLNetwordembeddingsin |     |     |     |
|                    | (cid:16) | (cid:17) |     |                                                     |     |     |     |
 ifc ∈д thelabeltext.Specifically,thelabelembeddingoflabell is
|     |  σ д(x,c | l ),h(x,l) , | l b (x), |     |     |     |     |
| --- | --------- | ------------ | -------- | --- | --- | --- | --- |
|     | f(x,l)=  |              |          | (2) |     |     |     |
|     | −∞,      | otherwise.   |          |     | 1   |     |     |
(cid:88)
|     |     |     |     | ψtext-emb(l)= |     | ϕxlnet(w) |     |
| --- | --- | --- | --- | ------------- | --- | --------- | --- |
Herec ∈{1,...,K}representstheclusterindexoflabell,д(x,c |text(l)|
| l   |     |     |     | l ) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
istheneuralmatcherrealizedbydeeppre-trainedTransformers, w∈text(l)
|     |     |     |     | whereϕ (w)isthehiddenembeddingoftokenwinlabell. |     |     |     |
| --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- |
h(x,l)isthelinearranker,andσ()isanon-linearactivationfunc- xlnet
| tion to combine | the final scores | fromд andh. | We now | further |     |     |     |
| --------------- | ---------------- | ----------- | ------ | ------- | --- | --- | --- |
Labelembeddingviaembeddingofpositiveinstances. The
introduceeachofthesethreecomponentsindetail.
shorttextoflabelsmaynotcontainsufficientinformationandis
oftenambiguousandnoisyforsomeXMCdatasets.Thereforewe
3.2 SemanticLabelIndexing
canderivealabelrepresentationfromembeddingofitspositive
Inducinglatentclusterswithsemanticmeaningbringsseveralad-
|     |     |     |     | instances.Specifically,thelabelembeddingoflabell |     |     | is  |
| --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- |
vantagestoourframework.Wecanperformaclusteringoflabels
(cid:88)
thatcanberepresentedbyalabel-to-clusterassignmentmatrix ψpifa-tfidf(l)=v /∥v ∥, v = ϕtf-idf(xi), l =1,...,L,
l l l
C∈{0,1}L×K wherec =1meanslabellbelongstoclusterk.The i:yil =1
lk
n u m b e r o f c l u s t e r sK i s t y pi c a ll y s e t to b e m u c h s m a l le r t h a n t h e (cid:88)
|     |     |     |     | ψpifa-neural(l)=v | /∥v ∥, v = | ϕxlnet(xi), | l =1,...,L. |
| --- | --- | --- | --- | ----------------- | ---------- | ----------- | ----------- |
or i gi n a ll a b e l s p a c e L. D e e p T r a n sf o r m er m o d e ls a r e fi n e -t u n e d o n l l l
i:yil =1
theinducedXMCsub-problemwheretheoutputspaceisofsizeK,
whichsignificantlyreducesthecomputationalcostandavoidsthe WerefertothistypeoflabelembeddingasPositiveInstanceFeature
labelsparsityissueinFigure1.Furthermore,thelabelclustering Aggregation(PIFA),whichisusedinrecentstate-of-the-artXMC
alsoplaysacrucialroleinthelinearrankerh(x,l).Forexample, methods[7,9,20,32].NotethatX-Transformerisnotlimitedby
onlylabelswithinaclusterareusedtoconstructnegativeinstances theabovementionedlabelrepresentations;indeedinapplications
fortrainingtheranker.Inprediction,rankingisonlyperformed wherelabelsencoderichermetainformationsuchasagraph,we
forlabelswithinafewclusterspredictedbyourdeepTransformer canuselabelrepresentationsderivedfromgraphclusteringand
| models. |     |     |     | graphconvolution. |     |     |     |
| ------- | --- | --- | --- | ----------------- | --- | --- | --- |
3166

Applied Data Science Track Paper   KDD '20, August 23–27, 2020, Virtual Event, USA
3.3 DeepTransformerasNeuralMatcher
AfterSemanticLabelIndexing(SLI),theoriginalintractableXMC
| problem                                         | morphs to a | feasible XMC | sub-problem | with | a much |     |     |     |     |     |
| ----------------------------------------------- | ----------- | ------------ | ----------- | ---- | ------ | --- | --- | --- | --- | --- |
| smalleroutputspaceofsizeK.SeeTable2fortheexactK |             |              |             |      | that   |     |     |     |     |     |
weusedforeachXMCdataset.Specifically,thedeepTransformer
| model now       | aims to map | each text           | instance      | to the assigned  | rel-      |     |     |     |     |     |
| --------------- | ----------- | ------------------- | ------------- | ---------------- | --------- | --- | --- | --- | --- | --- |
| evant clusters. | The induced | instance-to-cluster |               | assignment       | ma-       |     |     |     |     |     |
| tri x is M      | = Y C = [   | m 1 , . . . , m ,   | . . . , m ] T | ∈ { 0, 1 } N × K | w h e r e |     |     |     |     |     |
i N F i g u r e 4 : T r a i n in g r a n k e r s w it h th e Te ac he r F o r ci n g N eg a -
| Y ∈ R N ×L | ist h eo rig i                                  | na l i n s t a nc e | - t o - la b el a ss | ign m e n t m a | tr ix a n d |                 |                       |                          |                   |                   |
| ---------- | ----------------------------------------------- | ------------------- | -------------------- | --------------- | ----------- | --------------- | --------------------- | ------------------------ | ----------------- | ----------------- |
|            |                                                 |                     |                      |                 |             | t i v e s( T FN | ) st ra t e g y .F or | i l lu s t r ati on , we | h ave N = 6 in st | a n c e s,L = 2 0 |
| RL×K       | isthelabel-to-clusterassignmentmatrixprovidedby |                     |                      |                 |             |                 |                       |                          |                   |                   |
C ∈ labels,K =4labelclusters,andM∈ {0,1}6×4denotestheinstance-to-
theSLIstage.Thegoalnowbecomesfine-tuningdeepTransformer
clusterassignmentmatrix.Forexample,Cluster1withtheorange
modelsд(x,k;W,θ)on{(xi,mi)|i =1,...,N}suchthat colorcontainsthefirst5labels.Thenonzerosofthefirstcolumnof
|     |     |     |     |     |     | Mcorrespondto | {x1,x2,x6},whichareinstanceswithatleastone |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | ------------------------------------------ | --- | --- | --- |
(cid:88)N (cid:88)K
1 (cid:16) (cid:17)2 positivelabelcontainedinCluster1.Foreachlabelinthefirstclus-
| min |     | max 0,1−M | ˜ д(x,k;W,θ) | ,   | (3) |     |     |     |     |     |
| --- | --- | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
ik t e r , t h e ra n k e ru s in g T e a c he r F o r c i n g N eg a ti v e s ( T F N ) o nl y c o n si d -
| W,θ | NK i=1k=1 |     |     |     |     |                |                       |                             |                      |                     |
| --- | --------- | --- | --- | --- | --- | -------------- | --------------------- | --------------------------- | -------------------- | ------------------- |
|     |           |     |     |     |     | e r s t h e se | t h re e in s ta n ce | s . M a tc h e r - a w a re | N e g a t iv e s ( M | A N ) st r a te g y |
s.t. д(x,k;W,θ)=wTϕtransformer(x), isintroducedinSection3.4tofurtheraddimprovedhardnegatives
k
toenhancetheTFNstrategy.
˜
| whereM             | = 2M −1 | ∈ {−1,1},W                         | = [w1,...,wK | ]T  | ∈ RK×d, |     |     |     |     |     |
| ------------------ | ------- | ---------------------------------- | ------------ | --- | ------- | --- | --- | --- | --- | --- |
|                    | ik ik   |                                    |              |     |         |     |     |     |     |     |
| andϕtransformer(x) | ∈Rd     | istheembeddingfromtheTransformers. |              |     |         |     |     |     |     |     |
Weusethesquared-hingelossinthematchingobjective(3)asit
hasshownbetterrankingperformanceasshownin[31].Next,we Bootstrapping Label Clustering and Ranking. After fine-
tuningadeepTransformermodel,wehavepowerfulinstancerep-
discussengineeringoptimizationsandimplementationdetailsthat
resentationϕtransformer(x)thatcanbeusedtobootstrapsemantic
considerablyimprovetrainingefficiencyandmodelperformance.
labelclusteringandranking.Forlabelclustering,theembedding
Weconsiderthreestate-of-the-art labell canbeconstructedbyaggregatingtheembeddingsofits
PretrainedTransformers.
pre-trainedTransformer-large-casedmodels(i.e.,24layerswith positiveinstances.Forranking,thefine-tunedTransformerembed-
case-sensitivevocabulary)tofine-tune,namelyBERT[5],XLNet[30], dingcanbeconcatenatedwiththesparseTF-IDFvectorforbetter
andRoBERTa[13].Theinstanceembeddingϕ(x)isthe"[CLS]"-like modelingpower.SeedetailsintheablationstudyTable5.
hiddenstatesfromthelastlayerofBERT,RoBERTaandXLNet.
Computationallyspeaking,BERTandRoBERTaaresimilarwhile
|     |     |     |     |     |     | 3.4 Ranking |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- |
XLNetisnearly1.8timesslower.IntermsofperformanceonXMC
Afterthematchingstep,asmallsubsetoflabelclustersisretrieved.
| tasks, we | found RoBERTa | and XLNet | to be | slightly better | than |     |     |     |     |     |
| --------- | ------------- | --------- | ----- | --------------- | ---- | --- | --- | --- | --- | --- |
Thegoaloftherankeristomodeltherelevancebetweenthein-
BERT,butthegapisnotassignificantasintheGLUEbenchmark.
stanceandthelabelsfromtheretrievedclusters.Formally,givena
MoreconcreteanalysisisavailableinSection4.
labellandaninstancex,weusealinearone-vs-all(OVA)classifier
ItispossibletouseAutomaticMixedPrecision(AMP)between
Float32andFloat16formodelfine-tuning,whichcanconsiderably toparameterizetherankerh(x,l) = wT ϕ(x) andtrainitwitha
l
|                                                     |     |     |     |     |     | binaryloss.Foreachlabel,naivelyestimatingtheweightsw |     |                                     |     | based |
| --------------------------------------------------- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | ----------------------------------- | --- | ----- |
| reducethemodel’sGPUmemoryusageandtrainingspeed.How- |     |     |     |     |     |                                                      |     |                                     |     | l     |
|                                                     |     |     |     |     |     | onallinstances{(xi,Y                                 |     | )} N takesO(N),whichistooexpensive. |     |       |
ever,weusedFloat32foralltheexperimentsasourinitialtrialsof i,l i =1
Instead,weconsidertwosamplingstrategiesthatonlyincludehard
trainingTransformersinAMPmodeoftenledtounstablenumerical
negativeinstancestoreducethecomputationalcomplexity:Teacher
resultsforthelarge-scaleXMCdatasetWiki-500K.
ForcingNegatives(TFN)andMatcher-awareNegatives(MAN).
InputSequenceLength. Thetimeandspacecomplexityofthe TeacherForcingNegatives(TFN).foreachlabell,weonly
Transformerscalesquadraticallywiththeinputsequencelength, includeasubsetofinstancesinducedbytheinstance-to-cluster
i.e.,O(T 2 )[26],whereT =len(x)isthenumberoftokenizedsub- assignmentmatrixM = YC.Inparticular,inadditiontothepos-
wordsintheinstancex.UsingsmallerT reducesnotonlytheGPU itive instances corresponding to thel-th label, we only include
memoryusagethatsupportsusinglargerbatchsize,butalsoin- instanceswhoselabelsbelongtothesameclusterasthel-thlabel,
creasesthetrainingspeed.Forexample,BERTfirstpre-trainson i.e.,{(xi,y : i ∈ {i : Mi,cl = 1}}.InFigure4,weillustratethe
i,l
inputsofsequencelength128for90%oftheoptimization,andthe TFNstrategywithatoyexample.Asthefirstfivelabelsbelongto
remaining10%ofoptimizationstepsoninputsofsequencelength Cluster1,andonly{x1,x2,x6}containapositivelabelwithinthis
512[5].Interestingly,weobservethatthemodelfine-tunedwith cluster,weonlyconsiderthissubsetofinstancestotrainabinary
sequencelength128v.s.sequencelength512doesnotdiffersignif- classifierforeachofthefirstfivelabels.
icantlyinthedownstreamXMCrankingperformance.Thus,we Matcher-awareNegatives(MAN).TheTeacherForcingstrat-
fixtheinputsequencelengthtobeT =128formodelfine-tuning, egy only includes negative instances which are hard from the
whichsignificantlyspeedsupthetrainingtime.Itwouldbeinterest- “teacher”,i.e.,thegroundtruthinstance-to-clusteringassignment
ingtoseeifwecanbootstraptrainingtheTransformermodelsfrom matrixMusedtotrainourneuralmatcher.However,Misindepen-
shortersequencelengthandrampuptolargersequencelength dentfromtheperformanceofourneuralmatcher.Thus,training
(e.g.,32,64,128,256),butweleavethatasfuturework. rankerwiththeTFNstrategyalonemightintroduceanexposure
3167

Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
biasissue,i.e.,training-inferencediscrepancy.Instead,wealsocon- arelistedinTable2,whichareconsistentwiththeParabelsetting
siderincludingmatcher-awarehardnegativesforeachlabel.In forfaircomparison.Weconsiderthe24layerscasedmodelsof
particular, we can use the instance-to-cluster prediction matrix BERT[5],RoBERTa[13],andXLNet[30]usingthePytorchimple-
M ˆ ∈ {0,1}N×K fromourneuralmatcher,wherethenonzerosof mentationfromHuggingFaceTransformers[28]2.Forfine-tuning
thei-throwofM ˆ correspondtothetop-bpredictedclustersfrom theTransformermodels,wesettheinputsequencelengthtobe
д
b
(xi).Inpractice,weobservethatacombinationofTFNandMAN 128forefficiency,andthebatchsizeperGPUtobe16alongwith
yieldsthebestperformance,i.e.,usingM′ = YC+M ˆ toinclude gradientaccumulationstepof4,anduse4GPUspermodel.This
hardnegativesforeachlabel.SeeTable5foradetailedAblation togetheramountstoabatchsizeof256intotal.WeuseAdam[10]
study. withlinearwarmupschedulingastheoptimizerwherethelearn-
Fortherankerinputrepresentation,wenotonlyleveragethe ingrateischosenfrom{4,5,6,8}×10−5.Modelsaretraineduntil
TF-IDFfeaturesϕtf-idf(x),butalsoexploittheneuralembeddings convergence,whichtakes1k,1.4k,20k,50koptimizationstepsfor
ϕneural(x)fromeitherthepre-trainedorfine-tunedTransformer Eurlex-4K,Wiki10-31K,AmazonCat-13K,Wiki-500K,respectively.
model. After the ranker is trained, the final ranking scores are
computedvia(2).Wecanfurtherensemblethescoresfromdifferent 4.3 ResultsonPublicXMCBenchmarkData
X-Transformermodels,whicharetrainedondifferentsemantic- Table3comparestheproposedX-Transformerwiththemostrepre-
awarelabelclustersordifferentpre-trainedTransformermodels sentativeSOTAXMCmethodsonfourbenchmarkdatasets.Follow-
suchasBERT,RoBERTaandXLNet. ingpreviousXMCworks,wefocusontoppredictionsbypresenting
Precision@k,wherek =1,3,5.
4 EMPIRICALRESULTS TheproposedX-TransformeroutperformsallXMCmethods,ex-
Theexperimentcode,includingdatasetsandfine-tunedmodelsare ceptbeingslightlyworsethanAttentionXMLintermsofP@3and
publiclyavailable.1 P@5ontheWiki-500Kdataset.WealsocompareX-Transformer
againstlinearbaselinesusingParabelmodelwiththreedifferent
4.1 DatasetsandPreprocessing inputrepresentations:(1)ϕpre-xlnet denotespretrainedXLNetem-
XMCBenchmarkData. Weconsiderfourmulti-labeltextclas- beddings(2)ϕtfidf denotesTF-IDFembeddings(3)ϕfnt-xlnet⊕ϕtfidf
sificationdatasetsusedinAttentionXML[32]forwhichwehad denotesfinetunedXLNetembeddingsconcatenatedwithTF-IDF
accesstotherawtextrepresentation,namelyEurlex-4K,Wiki10- embeeddings.Weclearlyseethattheperformanceofbaseline(1)
31K,AmazonCat-13KandWiki-500K.Summarystatisticsofthe issignificantlyworse.ThissuggeststhattheELMo-styletransfer
datasetsaregiveninTable2.Wefollowthetrainingandtestsplit learning,thoughefficient,isnotpowerfultoachievegoodperfor-
of[32]andsetaside10%ofthetraininginstancesasthevalidation manceforXMCproblems.Theperformanceofbaseline(2)issimilar
setforhyperparametertuning. tothatofParabel,whilebaseline(3)furtherimprovesperformance
duetotheuseoffine-tunedXLNetembeddings.
AmazonApplications. WeconsideraninternalAmazondata AttentionXML[32]isaveryrecentdeeplearningmethodthat
set,namelyProd2Query-1M,whichconsistsof14millioninstances usesBiLSTMandlabel-awareattentionlayertomodelthescoring
(products)and1millionlabels(queries)wherethelabelispositive function.Theyalsoleveragehierarchicallabeltreestorecursively
ifaproductisclickedatleastonceasaresultofasearchquery.We warm-startthemodelsandusehardnegativesamplingtechniques
dividethedatasetinto12.5milliontrainingsamples,0.8million toavoidusingtheentireclassifierbottlenecklayer.Someofthe
validationsamplesand0.7milliontestingsamples. techniquesinAttentionXMLarecomplementarytoourproposedX-
Transformer,anditwouldbeinterestingtoseehowX-Transformer
4.2 AlgorithmsandHyperparameters canbeimprovedfromthosetechniques.
ComparingMethods. WecompareourproposedX-Transformer
methodtothemostrepresentativeandstate-of-the-artXMCmeth- 4.4 ResultsonAmazonApplications.
odsincludingtheembedding-basedAnnexML[24];one-versus-all
RecallthattheAmazondataconsistsof12millionproductsand
DiSMEC[1];instancetreebasedPfastreXML[8];labeltreebased
1millionqueriesalongwithproduct-queryrelevance.Wetreat
Parabel[20],eXtremeText[29],Bonsai[9];anddeeplearningbased
queriesasoutputlabelsandproducttitleasinput.Weusethedefault
XML-CNN[12],AttentionXML[32]methods.Theresultsofallthese
Parabelmethod(usingTFIDFfeatures)asthebaselinemethodand
baselinemethodsareobtainedfrom[32,Table3].Forevaluation
showX-Transformer’srelativeimprovementofprecisionandrecall
withotherXMCapproachesthathavenotreleasedtheircodeor
overthebaselineinTable4.
aredifficulttoreproduce,wehaveadetailedcomparisoninTable6.
EvaluationMetrics. Weevaluateallmethodswithexample- 4.5 AblationStudy
basedrankingmeasuresincludingPrecision@k(k = 1,3,5)and WecarefullyconductanablationstudyofX-Transformerasshown
Recall@k(k = 1,3,5),whicharewidelyusedintheXMClitera- inTable5.WeanalyzetheX-Transformerframeworkintermsofits
ture[3,8,20,21,23]. fourcomponents:indexing,matching,rankerinputrepresentation,
andtrainingnegative-samplingtrainingalgorithm.Theconfigu-
Hyperparameters. ForX-Transformer,allhyperparametersare
rationIndex9representsthefinalbestconfigurationasreported
chosenfromtheheld-outvalidationset.Thenumberofclusters
1https://github.com/OctoberChang/X-Transformer 2https://github.com/huggingface/transformers
3168

Applied Data Science Track Paper   KDD '20, August 23–27, 2020, Virtual Event, USA
|     |     | Dataset |        |       | |Dtrn|     | |Dtrn|    |     |       | ¯    | n¯    |     |
| --- | --- | ------- | ------ | ----- | ---------- | --------- | --- | ----- | ---- | ----- | --- |
|     |     |         | ntrn   | ntst  |            |           |     | L     | L    |       | K   |
|     |     |         | 15,449 | 3,865 | 19,166,707 | 4,741,799 |     | 3,956 | 5.30 | 20.79 | 64  |
Eurlex-4K
|     |     |     | 14,146 | 6,616 | 29,603,208 | 13,513,133 |     | 30,938 | 18.64 | 8.52 | 512 |
| --- | --- | --- | ------ | ----- | ---------- | ---------- | --- | ------ | ----- | ---- | --- |
Wiki10-31K
AmazonCat-13K 1,186,239 306,782 250,940,894 64,755,034 13,330 5.04 448.57 256
Wiki-500K 1,779,881 769,421 1,463,197,965 632,463,513 501,070 4.75 16.86 8192
Table2:DataStatistics.ntrn,ntst refertothenumberofinstancesinthetrainingandtestsets,respectively.|Dtrn |,|Dtst |refer
¯
tothenumberofwordtokensinthetrainingandtestcorpus,respectively.Listhenumberoflabels,Ltheaveragenumberof
labelsperinstance,n¯theaveragenumberofinstancesperlabel,andKisthenumberofclusters.Thefourbenchmarkdatasets
arethesameasAttentionXML[32]forfaircomparison.
|     |             | Methods   | Prec@1    | Prec@3 | Prec@5 | Methods     |     |            | Prec@1 | Prec@3 | Prec@5 |
| --- | ----------- | --------- | --------- | ------ | ------ | ----------- | --- | ---------- | ------ | ------ | ------ |
|     |             |           | Eurlex-4K |        |        |             |     | Wiki10-31K |        |        |        |
|     | AnnexML[24] |           | 79.66     | 64.94  | 53.52  | AnnexML[24] |     |            | 86.46  | 74.28  | 64.20  |
|     |             | DiSMEC[1] | 83.21     | 70.39  | 58.73  | DiSMEC[1]   |     |            | 84.13  | 74.72  | 65.94  |
PfastreXML[8] 73.14 60.16 50.54 PfastreXML[8] 83.57 68.61 59.10
|     |     | Parabel[20] | 82.12 | 68.91 | 57.89 | Parabel[20] |     |     | 84.19 | 72.46 | 63.37 |
| --- | --- | ----------- | ----- | ----- | ----- | ----------- | --- | --- | ----- | ----- | ----- |
eXtremeText[29] 79.17 66.80 56.09 eXtremeText[29] 83.66 73.28 64.51
|     |             | Bonsai[9] | 82.30 | 69.55 | 58.35 | Bonsai[9]   |     |     | 84.52 | 73.76 | 64.69 |
| --- | ----------- | --------- | ----- | ----- | ----- | ----------- | --- | --- | ----- | ----- | ----- |
|     | MLC2seq[16] |           | 62.77 | 59.06 | 51.32 | MLC2seq[16] |     |     | 80.79 | 58.59 | 54.66 |
|     | XML-CNN[12] |           | 75.32 | 60.14 | 49.21 | XML-CNN[12] |     |     | 81.41 | 66.23 | 56.11 |
AttentionXML[32] 87.12 73.99 61.92 AttentionXML[32] 87.47 78.48 69.37
ϕpre-xlnet +Parabel 33.53 26.71 22.15 ϕpre-xlnet +Parabel 81.77 64.86 54.49
|                   |        | +Parabel | 81.71 | 69.15 | 58.11             | +Parabel |          |     | 84.27 | 73.20 | 63.66 |
| ----------------- | ------ | -------- | ----- | ----- | ----------------- | -------- | -------- | --- | ----- | ----- | ----- |
|                   | ϕtfidf |          |       |       |                   | ϕtfidf   |          |     |       |       |       |
|                   |        | +Parabel | 84.09 | 71.50 | 60.12             |          | +Parabel |     | 87.35 | 78.24 | 68.62 |
| ϕfnt-xlnet⊕ϕtfidf |        |          |       |       | ϕfnt-xlnet⊕ϕtfidf |          |          |     |       |       |       |
X-Transformer 87.22 75.12 62.90 X-Transformer 88.51 78.71 69.62
|     |             | AmazonCat-13K |       |       |       |             |     | Wiki-500K |       |       |       |
| --- | ----------- | ------------- | ----- | ----- | ----- | ----------- | --- | --------- | ----- | ----- | ----- |
|     | AnnexML[24] |               | 93.54 | 78.36 | 63.30 | AnnexML[24] |     |           | 64.22 | 43.15 | 32.79 |
|     |             | DiSMEC[1]     | 93.81 | 79.08 | 64.06 | DiSMEC[1]   |     |           | 70.21 | 50.57 | 39.68 |
PfastreXML[8] 91.75 77.97 63.68 PfastreXML[8] 56.25 37.32 28.16
|     |     | Parabel[20] | 93.02 | 79.14 | 64.51 | Parabel[20] |     |     | 68.70 | 49.57 | 38.64 |
| --- | --- | ----------- | ----- | ----- | ----- | ----------- | --- | --- | ----- | ----- | ----- |
eXtremeText[29] 92.50 78.12 63.51 eXtremeText[29] 65.17 46.32 36.15
|     |             | Bonsai[9] | 92.98 | 79.13 | 64.46 | Bonsai[9]   |     |     | 69.26 | 49.80 | 38.83 |
| --- | ----------- | --------- | ----- | ----- | ----- | ----------- | --- | --- | ----- | ----- | ----- |
|     | MLC2seq[16] |           | 94.26 | 69.45 | 57.55 | MLC2seq[16] |     |     | -     |       | - -   |
|     | XML-CNN[12] |           | 93.26 | 77.06 | 61.40 | XML-CNN[12] |     |     | -     |       | - -   |
AttentionXML[32] 95.92 82.41 67.31 AttentionXML[32] 76.95 58.42 46.14
ϕpre-xlnet +Parabel 80.96 63.92 50.72 ϕpre-xlnet +Parabel 31.83 20.24 15.76
|                   |        | +Parabel | 92.81 | 78.99 | 64.31             | +Parabel |          |     | 68.75 | 49.54 | 38.92 |
| ----------------- | ------ | -------- | ----- | ----- | ----------------- | -------- | -------- | --- | ----- | ----- | ----- |
|                   | ϕtfidf |          |       |       |                   | ϕtfidf   |          |     |       |       |       |
|                   |        | +Parabel | 95.33 | 82.77 | 67.66             |          | +Parabel |     | 75.57 | 55.12 | 43.31 |
| ϕfnt-xlnet⊕ϕtfidf |        |          |       |       | ϕfnt-xlnet⊕ϕtfidf |          |          |     |       |       |       |
X-Transformer 96.70 83.85 68.58 X-Transformer 77.28 57.47 45.31
Table3:ComparingX-Transformeragainststate-of-the-artXMCmethodsonEurlex-4K,Wiki10-31K,AmazonCat-13K,andWiki-500K.
Thebaselines’resultsarefrom[32,Table3].NotethatMLC2seqandXML-CNNarenotscalableonWiki-500K.Wealsopresent
linearbaselines(Parabel)withthreeinputrepresentations.Specifically,ϕ denotespre-trainedXLNetembeddings,ϕ
pre-xlnet tfidf
⊕ϕ
denotesTF-IDFembeddings,ϕ fnt-xlnet tfidf denotesfine-tunedXLNetembeddingsconcatenatewithTF-IDFembeddings.
Precision Recall inTable3.Therearefourtakeawaymessagesfromthisablation
Methods @1 @5 @10 @1 @5 @10 study,andwedescribetheminthefollowingfourparagraphs.
X-Transformer 10.7% 7.4% 6.6% 12.0% 4.9% 2.8% RankerRepresentationandTraining. Config.ID0,1,2shows
theeffectofinputrepresentationandtrainingstrategyfortherank-
| Table 4: | Relative | improvement | over | Parabel | on the |     |     |     |     |     |     |
| -------- | -------- | ----------- | ---- | ------- | ------ | --- | --- | --- | --- | --- | --- |
ing.Thebenefitofusinginstanceembeddingfromfine-tunedtrans-
Prod2Querydataset.
formerscanbeseenfromconfig.ID0to1.Inaddition,fromID1
to2,weobservethatusingTeacherForcingNegatives(TFN)isnot
enoughfortrainingtheranker,asitcouldsufferfromtheexposure
3169

Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
X-TransformerAblationConfiguration EvaluationMetric
Dataset Config.ID
indexing matching rankerinput negative-sampling P@1 P@3 P@5 R@1 R@3 R@5
0 pifa-tfidf BERT ϕtfidf(x) TFN 83.93 70.59 58.69 17.05 42.08 57.14
1 pifa-tfidf BERT ϕtfidf(x)⊕ϕneural(x) TFN 85.02 71.83 59.87 17.21 42.79 58.30
2 pifa-tfidf BERT ϕtfidf(x)⊕ϕneural(x) TFN+MAN 85.51 72.95 60.83 17.32 43.45 59.21
3 pifa-tfidf RoBERTa ϕtfidf(x)⊕ϕneural(x) TFN+MAN 85.33 72.89 60.79 17.32 43.39 59.16
4 pifa-tfidf XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 85.07 72.75 60.69 17.25 43.29 59.01
Eurlex-4K
5 pifa-neural XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 84.81 72.39 60.38 17.19 42.98 58.70
6 text-emb XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 85.25 72.76 60.20 17.29 43.25 58.54
7 all XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 86.55 74.24 61.96 17.54 44.16 60.24
8 pifa-neural all ϕtfidf(x)⊕ϕneural(x) TFN+MAN 85.92 73.43 61.53 17.40 43.69 59.86
9 all all ϕtfidf(x)⊕ϕneural(x) TFN+MAN 87.22 75.12 62.90 17.69 44.73 61.17
0 pifa-tfidf BERT ϕtfidf(x) TFN 69.52 49.87 38.71 22.30 40.62 48.65
1 pifa-tfidf BERT ϕtfidf(x)⊕ϕneural(x) TFN 71.90 51.58 40.10 23.27 42.14 50.42
2 pifa-tfidf BERT ϕtfidf(x)⊕ϕneural(x) TFN+MAN 74.68 53.64 41.50 24.56 44.26 52.50
3 pifa-tfidf RoBERTa ϕtfidf(x)⊕ϕneural(x) TFN+MAN 75.40 54.32 42.06 24.85 44.93 53.30
4 pifa-tfidf XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 75.45 54.50 42.24 24.81 45.00 53.44
Wiki-500K
5 pifa-neural XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 76.34 55.50 43.04 25.15 45.88 54.53
6 text-emb XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 74.12 52.85 40.53 24.18 43.30 50.98
7 all XLNet ϕtfidf(x)⊕ϕneural(x) TFN+MAN 75.85 56.08 44.24 24.80 46.36 56.35
8 pifa-neural all ϕtfidf(x)⊕ϕneural(x) TFN+MAN 77.44 56.84 44.37 25.61 47.18 56.55
9 all all ϕtfidf(x)⊕ϕneural(x) TFN+MAN 77.28 57.47 45.31 25.48 47.82 57.95
Table5:AblationstudyofX-TransformeronEurlex-4KandWiki-500Kdatasets.Weoutlinefourtakeawaymessages:(1)Config.
ID= {0,1,2}demonstratesbetterperformancebyusingMatcher-awareNegatives(MAN)andNeuralembeddingfortraining
therankers;(2)Config.ID= {2,3,4}suggeststhat,performance-wise,XLNetissimilartoRoBERTa,andslightlybetterthan
BERT;(3)Config.ID={4,5,6}manifeststheimportanceoflabelclustersinducedfromdifferentlabelrepresentations.(4)Config.
ID={7,8,9}indicatestheeffectofensemblingvariousconfigurationofthemodels.
biasofonlyusingthegroundtruthclusteringassignment,butig- 4.6 Cross-PaperComparisons
noresthehardnegativesmistakenlyproducedbytheTransformer ManyXMCapproacheshavebeenproposedrecently.However,it
models.NotethattechniquessuchasaddingMatcher-awareneg- issometimesdifficulttocomparemetricsdirectlyfromdifferentpa-
atives(MAN)frompreviousmodel’spredictiontobootstrapthe pers.Forexample,theP@1ofParabelonWiki-500Kis59.34%in[7,
nextlevel’smodeltrainingisalsousedinAttentionXML[32]. Table2]and68.52%in[20,Table2],butwesee68.70%inTable3.
Theinconsistencymaybeduetodifferencesindataprocessing,
Different Transformer Models. Next, we analyze how the inputrepresentation,orotherreasons.Weproposeanapproachto
threedifferentTransformermodels(i.e.,BERT,RoBERTa,XLNet) calibratethesenumberssothatvariousmethodscanbecompared
affecttheperformance,asshowninConfig.ID2,3,4.ForWiki- inamoreprincipledway.Inparticular,foreachmetricm(·),weuse
500K,weobservethattheXLNetandRoBERTaaregenerallymore therelativeimprovementoveracommonanchormethod,whichis
powerfulthantheBERTmodels.Ontheotherhand,suchanad- settobeParabelasitiswidelyusedintheliterature.Foracompet-
vantageisnotclearforEurlex-4K,possiblyduetothenatureofthe ingmethodXwithametricm(X)onadatasetreportedinapaper,
dataset. wecancomputetherelativeimprovementoverParabelasfollows:
m(X)−m(Parabel) ×100%,wherem(Parabel)isthemetricobtained
LabelRepresentationforClustering. Theimportanceofdif- byP m a ( r P a ar b a e b l el o ) nthesamedatasetinthesamepaper.Followingthe
ferentlabelrepresentationforclusteringisdemonstratedinConfig.
aboveapproach,weincludeavarietyofXMCapproachesinour
ID4,5,6.ForEurlex-4K,weseethatusinglabeltextembedding
comparison.Wereporttherelativeimprovementofvariousmeth-
asrepresentation(i.e.text-emb)leadstothestrongperformance
odsontwocommonlyuseddatasets,Eurlex-4KandWiki-500K,in
comparedtopifa-tfidf(id4)andpifa-neural(id5).Incontrast,pifa-
Table6.WecanclearlyobservethatX-Transformerbringsthemost
tfidfbecomesthebestperformingrepresentationontheWiki-500K
significantimprovementoverParabelandSLICE.
dataset.ThisphenomenoncouldbeduetothelabeltextofWiki-
500KbeingmorenoisycomparedtoEurlex-4K,whichdeteriorates
thelabelclusteringresultsonWiki-500K. 5 CONCLUSIONS
Inthispaper,weproposeX-Transformer,thefirstscalableframe-
EnsembleRanking. Finally,weshowtheadvantageofensem- worktofine-tuneDeepTransformermodelsthatimprovesstate-of-
bingpredictionfromdifferentmodelsasshowninConfig.ID7,8,9. the-artXMCmethodsonfourXMCbenchmarkdatasets.Wefur-
ForEurlex-4K,combiningpredictionsfromdifferentlabelrepresen- therappliedX-Transformertoareal-lifeapplication,product2query
tations(ID7)isbetterthanfromdifferentTransformermodels(ID prediction,showingsignificantimprovementoverthecompetitive
8).Combiningall(ID9)leadstoourfinalmodel,X-Transformer. linearmodels,Parabel.
3170

Applied Data Science Track Paper KDD '20, August 23–27, 2020, Virtual Event, USA
Eurlex-4K Wiki-500K
RelativeImprovement RelativeImprovement
Method Source overParabel(%) Method Source overParabel(%)
Prec@1 Prec@3 Prec@5 Prec@1 Prec@3 Prec@5
X-Transformer Table3 +6.27% +9.08% +8.55% X-Transformer Table3 +12.49% +15.94% +17.26%
SLICE [7,Table2] +4.27% +3.34% +3.11% SLICE [7,Table2] +5.53% +7.02% +7.56%
GLaS [6,Table3] -5.18% -5.48% -5.34% GLaS [6,Table3] +4.77% +3.37% +4.27%
ProXML [2,Table5] +3.86% +2.90% +2.43% ProXML [2,Table5] +2.22% +0.82% +2.92%
PPD-Sparse [20,Table2] +1.92% +2.93% +2.92% PPD-Sparse [20,Table2] +2.39% +2.33% +2.88%
SLEEC [9,Table2] -3.53% -6.40% -9.04% SLEEC [9,Table2] -29.84% -40.73% -45.08%
Table6:ComparisonofRelativeImprovementoverParabel.Therelativeimprovementforeachstate-of-the-art(SOTA)method
iscomputedbasedonthemetricsreportedfromitsoriginalpaperasdenotedintheSourcecolumn.
REFERENCES [18] JeffreyPennington,RichardSocher,andChristopherDManning.2014.Glove:
[1] RohitBabbarandBernhardSchölkopf.2017. DiSMEC:distributedsparsema- Globalvectorsforwordrepresentation.InEMNLP.1532–1543.
chinesforextrememulti-labelclassification.InWSDM. [19] MatthewEPeters,MarkNeumann,MohitIyyer,MattGardner,Christopher
[2] RohitBabbarandBernhardSchölkopf.2019. Datascarcity,robustnessand Clark,KentonLee,andLukeZettlemoyer.2018. Deepcontextualizedword
extrememulti-labelclassification.MachineLearning(2019),1–23. representations.InProceedingsofthe2018ConferenceoftheNorthAmerican
[3] KushBhatia,HimanshuJain,PurushottamKar,ManikVarma,andPrateekJain. ChapteroftheAssociationforComputationalLinguistics(NAACL).
2015.Sparselocalembeddingsforextrememulti-labelclassification.InNIPS. [20] YashotejaPrabhu,AnilKag,ShrutendraHarsola,RahulAgrawal,andManik
[4] Wei-ChengChang,FelixX.Yu,Yin-WenChang,YimingYang,andSanjivKu- Varma.2018. Parabel:Partitionedlabeltreesforextremeclassificationwith
mar.2020. Pre-trainingTasksforEmbedding-basedLarge-scaleRetrieval.In applicationtodynamicsearchadvertising.InWWW.
InternationalConferenceonLearningRepresentations. [21] YashotejaPrabhuandManikVarma.2014.Fastxml:Afast,accurateandstable
[5] JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova.2019.Bert: tree-classifierforextrememulti-labellearning.InKDD.
Pre-trainingofdeepbidirectionaltransformersforlanguageunderstanding.In [22] AlecRadford,KarthikNarasimhan,TimSalimans,andIlyaSutskever.2018.Im-
provinglanguageunderstandingbygenerativepre-training.(2018).
Proceedingsofthe2019ConferenceoftheNorthAmericanChapteroftheAssociation
forComputationalLinguistics(NAACL). [23] SashankJReddi,SatyenKale,FelixYu,DanHoltmann-Rice,JiecaoChen,and
[6] ChuanGuo,AliMousavi,XiangWu,DanielNHoltmann-Rice,SatyenKale, SanjivKumar.2019.StochasticNegativeMiningforLearningwithLargeOutput
Sashank Reddi, and Sanjiv Kumar. 2019. Breaking the Glass Ceiling for Spaces.InAISTATS.
Embedding-BasedClassifiersforLargeOutputSpaces.InAdvancesinNeural [24] YukihiroTagami.2017. AnnexML:Approximatenearestneighborsearchfor
InformationProcessingSystems.4944–4954. extrememulti-labelclassification.InProceedingsofthe23rdACMSIGKDDinter-
[7] HimanshuJain,VenkateshBalasubramanian,BhanuChunduri,andManikVarma. nationalconferenceonknowledgediscoveryanddatamining.455–464.
2019.Slice:ScalableLinearExtremeClassifiersTrainedon100MillionLabelsfor [25] ManikVarma.2019.TheExtremeClassificationRepository:Multi-labelDatasets
RelatedSearches.InProceedingsoftheTwelfthACMInternationalConferenceon &Code.http://manikvarma.org/downloads/XC/XMLRepository.html.
WebSearchandDataMining.ACM,528–536. [26] AshishVaswani,NoamShazeer,NikiParmar,JakobUszkoreit,LlionJones,
[8] HimanshuJain,YashotejaPrabhu,andManikVarma.2016. Extrememulti- AidanNGomez,ŁukaszKaiser,andIlliaPolosukhin.2017. Attentionisall
labellossfunctionsforrecommendation,tagging,ranking&othermissinglabel youneed.InNIPS.
applications.InKDD. [27] AlexWang,AmanpreetSingh,JulianMichael,FelixHill,OmerLevy,andSamuelR
[9] SujayKhandagale,HanXiao,andRohitBabbar.2019.Bonsai-DiverseandShallow Bowman.2018.Glue:Amulti-taskbenchmarkandanalysisplatformfornatural
TreesforExtremeMulti-labelClassification. arXivpreprintarXiv:1904.08249 languageunderstanding.arXivpreprintarXiv:1804.07461(2018).
(2019). [28] ThomasWolf,LysandreDebut,VictorSanh,JulienChaumond,ClementDelangue,
[10] DiederikKingmaandJimmyBa.2014.Adam:Amethodforstochasticoptimiza- AnthonyMoi,PierricCistac,TimRault,R’emiLouf,MorganFuntowicz,andJamie
tion.InProceedingsoftheInternationalConferenceonLearningRepresentations. Brew.2019. HuggingFace’sTransformers:State-of-the-artNaturalLanguage
[11] KentonLee,Ming-WeiChang,andKristinaToutanova.2019.Latentretrievalfor Processing.ArXivabs/1910.03771(2019).
weaklysupervisedopendomainquestionanswering.InProceedingsofthe57th [29] MarekWydmuch,KalinaJasinska,MikhailKuznetsov,RóbertBusa-Fekete,and
AnnualMeetingoftheAssociationforComputationalLinguistics(ACL). KrzysztofDembczynski.2018.Ano-regretgeneralizationofhierarchicalsoftmax
[12] JingzhouLiu,Wei-ChengChang,YuexinWu,andYimingYang.2017. Deep toextrememulti-labelclassification.InNIPS.
learningforextrememulti-labeltextclassification.InProceedingsofthe40th [30] ZhilinYang,ZihangDai,YimingYang,JaimeCarbonell,RuslanSalakhutdinov,
andQuocVLe.2019.XLNet:GeneralizedAutoregressivePretrainingforLan-
InternationalACMSIGIRConferenceonResearchandDevelopmentinInformation
Retrieval.ACM,115–124. guageUnderstanding.InNIPS.
[13] YinhanLiu,MyleOtt,NamanGoyal,JingfeiDu,MandarJoshi,DanqiChen,Omer [31] IanEHYen,XiangruHuang,WeiDai,PradeepRavikumar,InderjitDhillon,and
Levy,MikeLewis,LukeZettlemoyer,andVeselinStoyanov.2019.RoBERTa:A EricXing.2017.PPDsparse:Aparallelprimal-dualsparsemethodforextreme
RobustlyOptimizedBERTPretrainingApproach.arXivpreprintarXiv:1907.11692 classification.InKDD.ACM.
(2019). [32] RonghuiYou,ZihanZhang,ZiyeWang,SuyangDai,HiroshiMamitsuka,and
[14] MikkoIMalinenandPasiFränti.2014.Balancedk-meansforclustering.InJoint ShanfengZhu.2019.AttentionXML:LabelTree-basedAttention-AwareDeep
IAPRInternationalWorkshopsonStatisticalTechniquesinPatternRecognition(SPR)
ModelforHigh-PerformanceExtremeMulti-LabelTextClassification.InAd-
andStructuralandSyntacticPatternRecognition(SSPR).Springer,32–41. vancesinNeuralInformationProcessingSystems.5812–5822.
[15] TomasMikolov,IlyaSutskever,KaiChen,GregSCorrado,andJeffDean.2013.
Distributedrepresentationsofwordsandphrasesandtheircompositionality.In
Advancesinneuralinformationprocessingsystems.3111–3119.
[16] JinseokNam,EneldoLozaMencía,HyunwooJKim,andJohannesFürnkranz.
2017.MaximizingSubsetAccuracywithRecurrentNeuralNetworksinMulti-
labelClassification.InNIPS.
[17] IoannisPartalas,ArisKosmopoulos,NicolasBaskiotis,ThierryArtieres,George
Paliouras,EricGaussier,IonAndroutsopoulos,Massih-RezaAmini,andPatrick
Galinari.2015.LSHTC:Abenchmarkforlarge-scaletextclassification.arXiv
preprintarXiv:1503.08581(2015).
3171
