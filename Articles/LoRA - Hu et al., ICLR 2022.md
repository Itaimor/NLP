<!-- page 1 -->

LoRA: Low-Rank Adaptation of
Large Language Models
EdwardHu∗ YelongShen∗ PhillipWallis
ZeyuanAllen-Zhu YuanzhiLi SheanWang WeizhuChen
MicrosoftCorporation
{edwardhu, yeshe, phwallis, zeyuana, yuanzhil
swang, wzchen}@microsoft.com
Abstract
The dominant paradigm of natural language processing consists of large-scale
pre-trainingongeneraldomaindataandadaptationtoparticulartasksordomains.
Aswepre-trainlargermodels,conventionalfine-tuning,whichretrainsallmodel
parameters,becomeslessfeasible. UsingGPT-3175Basanexample,deploying
manyindependentinstancesoffine-tunedmodels,eachwith175Bparameters,is
extremelyexpensive. WeproposeLow-RankAdaptation,orLoRA,whichfreezes
thepre-trainedmodelweightsandinjectstrainablerankdecompositionmatrices
intoeachlayeroftheTransformerarchitecture,greatlyreducingthenumberof
trainable parameters for downstream tasks. For GPT-3, LoRA can reduce the
numberoftrainableparametersby10,000timesandthecomputationhardware
requirement by 3 times compared to full fine-tuning. LoRA performs on-par
or better than fine-tuning in model quality on both GPT-3 and GPT-2, despite
havingfewertrainableparameters,ahighertrainingthroughput,andnoadditional
inferencelatency.Wealsoprovideanempiricalinvestigationintorank-deficiencyin
languagemodeladaptations,whichshedslightontheefficacyofLoRA.Werelease
ourimplementationinGPT-2athttps://github.com/microsoft/LoRA.
1 Introduction
Manyapplicationsinnaturallanguageprocessingrelyonadapting
f(x)
onelargescale,pre-trainedlanguagemodeltomultipledownstream h
applications. Suchadaptationisusuallydoneviafine-tuning,which
updates all the parameters of the pre-trained model. The major
Pretrained
downsideoffine-tuningisthatitrequiresstoringasmanyparameters Pretrained 𝐵=0 Weights
as in the original model. As larger models are trained every few Weights 𝑟
months,thischangesfromamere“inconvenience”forGPT-2[32] 𝑊∈ℝ𝑑×𝑑
orBERT-large[9]toacriticaldeploymentchallengeforGPT-3[6]
𝑊∈ℝ𝑑×𝑑
𝐴=𝒩(0,𝜎2)
with175billiontrainableparameters.2 𝑑
𝑑
Many researchers sought to mitigate this by adapting only some x
x
parametersorlearningexternalmodulesfornewtasks. Thisway,we
Figure 1: Our reparametriza-
onlyneedtoloadasmallnumberoftask-specificparameterstothe
tion. WeonlytrainAandB.
pre-trainedmodelforeachtask,whichgreatlybooststhedeployment
efficiency. However,inpractice,existingtechniqueseitherintroduce
inferencelatency[15,34]byextendingmodeldepthorreducethe
∗Equalcontribution.
2WhileGPT-3175Bachievesnon-trivialperformancewithfew-shotlearning,fine-tuningboostsitsperfor-
mancesignificantlyasshowninAppendixA.
Preprint.Underreview.
1202
nuJ
71
]LC.sc[
1v58690.6012:viXra

<!-- page 2 -->

model’susablesequencelength[14,19,21,25]. Moreimportantly,thesepriorattemptssometimes
failtomatchthefine-tuningbaselines,posingatrade-offbetweenefficiencyandmodelquality.
Wetakeinspirationfrom[1,20]whichshowthatthelearnedover-parametrizedmodelsinfactreside
onalowintrinsicdimension. Wehypothesizethattheupdatematricesinlanguagemodeladaptation
alsohavealow“intrinsicrank”,leadingtoourproposedLow-RankAdaptation(LoRA)approach.
LoRAallowsustotraineverydenselayerinaneuralnetworkindirectlybyinjectingandoptimizing
rankdecompositionmatricesofthedenselayer’supdateinstead,whilekeepingtheoriginalmatrices
frozen,asshowninFig.1. UsingGPT-3175Basanexample,weshowthataverylowrank(i.e.,rin
Fig.1canbeoneortwo)sufficesevenwhenthefullrank(i.e.,d)isashighas12288,makingLoRA
bothspace-andcompute-efficient.
LoRApossessesseveralkeyadvantagesasfollows.
• A single pre-trained model can be shared and used to build many small LoRA modules for
differenttasks. WecankeepthesharedoriginalmodelinVRAMandefficientlyswitchtasksby
replacingthematricesAandBinFig.1,whichsignificantlyreducesthestoragerequirementand
task-switchingoverhead.
• Itmakestrainingmoreefficientandlowersthehardwarebarriertoentryby3times,sincewedo
notneedtocalculatethegradientsormaintaintheoptimizerstatesformostmodelparameters
whenusingadaptiveoptimizers. Instead,weonlyoptimizetheinjectedlow-rankmatrices,which
havemuchfewerparameters.
• Itssimplelineardesignallowsustomergetheupdatematriceswiththeoriginalweightsduring
deployment,introducingnoinferencelatency.
• LoRA is orthogonal to prior techniques and can be combined with many of them (such as
prefix-tuning). WeprovideanexampleinAppendixD.
Terminologies WemakefrequentreferencestotheTransformerarchitectureandusetheconven-
tionalterminologiesforitsdimensions. Wecallitshiddensize,orthesizeofitsactivations,d .
model
We use W , W , W , and W to refer to the query/key/value/output projection matrices in the
q k v o
self-attentionmodule. W orW referstoapre-trainedweightmatrixand∆W itsupdateduring
0
adaptation. WeusertodenotetherankofaLoRAmodule.
2 ProblemStatement
While our proposal is agnostic to the training objective, we focus on language modeling as the
primaryusecase. Belowisabriefdescriptionofthelanguagemodelingproblemand,inparticular,
themaximizationofconditionalprobabilitiesgivenatask-specificprompt.
Suppose we are given a pre-trained autoregressive language model p (y|x) that is parametrized
Φ
by Φ. For instance, p (y|x) can be a generic multi-task learner such as GPT-2 [32] or GPT-3
Φ
[6]basedontheTransformerarchitecture[36]. Now,consideradaptingthispre-trainedmodelto
(possiblymultiple)downstreamconditionaltextgenerationtasks,suchassummarization,machine
readingcomprehension(MRC),andnaturallanguagetoSQL(NL2SQL).Eachdownstreamtask
isrepresentedbyatrainingdatasetofcontext-targetpairs: Z = {(x ,y )} ,wherebothx
i i i=1,..,N i
andy aresequencesoftokens. Forexample,inNL2SQL,x isanaturallanguagequeryandy its
i i i
equivalentSQLcommand;forsummarization,x isthecontentofanarticleandy itsshortsummary.
i i
Duringfine-tuning,themodelisinitializedwithpre-trainedparametersΦ andupdatedtoΦ +∆Φ
0 0
byrepeatedlyfollowingthegradienttomaximizetheconditionallanguagemodelingobjective:
|y|
(cid:88) (cid:88)
max log(p (y |x,y )) (1)
Φ t <t
Φ
(x,y)∈Zt=1
Oneofthemaindrawbacksforfullfine-tuningisthatforeachdownstreamtask,welearnadifferent
setofparameters∆Φwhosedimensions|∆Φ|equals|Φ |. Thus,ifthepre-trainedmodelislarge
0
(such as GPT-3 with |Φ | ≈ 175Billion), storing and deploying many independent instances of
0
fine-tunedmodelscanbechallenging,ifatallplausible.
2

<!-- page 3 -->

Inthispaper,weadoptaparameter-efficientapproach,wherethetask-specificparameterincrement
∆Φ=∆Φ(Θ)isfurtherencodedbyamuchsmaller-sizedsetofparametersΘwith|Θ|(cid:28)|Φ |. The
0
taskoffinding∆ΦthusbecomesoptimizingoverΘ:
|y|
(cid:88) (cid:88) (cid:0) (cid:1)
max log p (y |x,y ) (2)
Θ
Φ0+∆Φ(Θ) t <t
(x,y)∈Zt=1
Asweshallseeinthesubsequentsections,weproposetousealow-rankrepresentationtoencode
∆Φthatisbothcomputationalandmemoryefficient. Whenthepre-trainedmodelisGPT-3,thesize
oftrainableparameters|Θ|canbeassmallas0.01%of|Φ |.
0
3 OurMethod
WedescribethesimpledesignofLoRAanditspracticalimplications. Theprinciplesoutlinedhere
apply to any dense layers in deep learning models, though we only focus on certain weights in
Transformersinourexperimentsforpracticalreasons.
Low-RankConstraintonUpdateMatrices. Atypicalneuralnetworkcontainsnumerousdense
layersthatperformmatrixmultiplication. Theweightmatricesintheselayersareallowedtohave
full-rank. When adapting to a specific task, however, Aghajanyan et al. [1] shows that the pre-
trainedlanguagemodelshavealow“instrisicdimension”andcanstilllearnefficientlydespitea
low-dimensionalreparametrization. Inspiredbythisobservation,wewonderiftheupdatestothe
weights also have a low “intrinsic rank" when adapting to downstream tasks. For a pre-trained
weightmatrixW ∈Rd×k,weconstrainitsupdatebyrepresentingitwithalow-rankdecomposition
0
W +∆W =W +BA,whereB ∈Rd×r,A∈Rr×k,andtherankr (cid:28)min(d,k).Duringtraining,
0 0
W isfrozenanddoesnotreceivegradientupdates, whileAandB containtrainableparameters.
0
NotebothW and∆W =BAaremultipliedwiththesameinput,andtheirrespectiveoutputvectors
0
aresummedcoordinate-wise. Forh=W x,ourmodifiedforwardpassyields:
0
h=W x+∆Wx=W x+BAx (3)
0 0
WeillustrateourreparametrizationinFig.1. WeusearandomGaussianinitializationforAandzero
forB,so∆W =BAiszeroatthebeginningoftraining. Wethenscale∆Wxby 1 duringtraining
r
tokeepthecoordinatesof∆WxroughlyΘ(1)inraftertraining[? ].
WeightDecaytoPre-trainedWeights. WenotethatweightdecaybehavesdifferentlywithLoRA
thanwithfullfine-tuning. Specifically,preformingtheusualweightdecayonAandBissimilarto
decayingbacktothepre-trainedweights,whichhasbeenstudiedasapotentiallyeffectiveformof
regularizationagainst“catastrophicforgetting”[16,18]. Whileextensiveexperimentsisolatingits
effectisout-of-scopeforthiswork,webelievethatthis,coupledwithaconstrainedparameterspace,
mightprovidesomeregularizationadvantages. Forexample,seehowLoRAwithr =d =1024
model
outperformsfullfine-tuningonGPT-2MediuminSec.5.3andSec.G.2.
NoAdditionalInferenceLatency. Duringdeployment,wecanexplicitlycomputeW =W +BA
0
andperforminferenceasusual. Whenweneedtoswitchtoanotherdownstreamtask,wecanrecover
W by subtracting BA and then adding a different B(cid:48)A(cid:48). This causes a minor increase in peak
0
memoryusageandaddsalatencytomodelswitchingthatdoesnotexceedasinglemodelforward
pass. Critically,wedonotintroduceanyadditionallatencyduringinferenceinreturn.
3.1 ApplyingLoRAtoTransformer
Inprinciple,wecanapplyLoRAtoanysubsetofweightmatricesinaneuralnetworktoreducethe
numberoftrainableparameters. IntheTransformerarchitecture,therearefourweightmatricesinthe
self-attentionmodule(W ,W ,W ,W )andtwointheMLPmodule. WetreatW (orW ,W )as
q k v o q k v
asinglematrixofdimensiond ×d ,eventhoughtheoutputdimensionisusuallysliced
model model
intoattentionheads. Welimitourstudytoonlychangingtheattentionweightsfordownstream
tasksandfreezetheMLPmodules(sotheyarenottrainedindownstreamtasks)sinceapplyingLoRA
tothelatterresultsin4timesthenumberoftrainableparametersgiventhesamerankr. Wefurther
studytheeffectonadaptingdifferenttypesofweightmatricesinaTransformerinSec.6.1.
3

<!-- page 4 -->

Practical Benefits and Limitations. The most significant benefit comes from the reduction in
memory and storage usage. For a Transformer, we reduce the VRAM consumption by 2/3 if
r (cid:28)d aswedonotneedtokeeptrackoftheoptimizerstatesforthefrozenparameters. Wealso
model
reducethecheckpointsizebyroughly dmodel times,whereγ isthefractionofweightmatriceson
2γr
whichweapplyLoRA.OnGPT-3,ourmotivatingusecase,wereducetheVRAMconsumptionfrom
1.2TBto350GB.Withr =4andγ =1/6,thecheckpointsizeisreducedbyroughly10,000×(from
350GBto35MB)3.ThisallowsustotrainwithsignificantlyfewerGPUsandavoidI/Obottlenecks.
Anotherbenefitisthatduringdeployment,wecanswitchbetweentasksatamuchlowercostbyonly
swappingtheLoRAweights,oftenmeasuredinmegabytes,asopposedtoalltheweights(350GB).
Thisallowsforthecreationofmanycustomizedmodelsthatcanbeactivatedanddeactivatedonthe
flyonmachinesthatstorethepre-trainedweights. Wealsoobservea25%speedupduringtrainingas
wedonotneedtocalculatethegradientforthevastmajorityoftheparameters.
Ontheotherhand,LoRAhasitslimitations. Forexample,itisnotstraightforwardtobatchinputsto
differenttaskswithdifferentAandBinasingleforwardpass,becauseweabsorbAandBintoW
topreventadditionalinferencelatency.
4 RelatedWorks
Transformer Language Models. Transformer [36] is a sequence-to-sequence architecture that
makesheavyuseofself-attention. [31]appliedittoautoregressivelanguagemodelingbyusinga
stackofTransformerdecoders. Sincethen,Transformer-basedlanguagemodelshavedominated
NLP,achievingthestate-of-the-artinmanytasks. AnewparadigmemergedwithBERT[9]and
GPT-2[32]–botharelargeTransformerlanguagemodelstrainedonalargeamountoftext–where
fine-tuning on task-specific data after pre-training on general domain data provides a significant
performancegaincomparedtotrainingontask-specificdatadirectly. TraininglargerTransformers
generallyresultsinbetterperformanceandremainsanactiveresearchdirection. GPT-3[6]isthe
largestsingleTransformerlanguagemodeltrainedto-datewith175Bparameters.
PromptEngineeringandFine-Tuning. WhileGPT-3175Bcanadaptitsbehaviorwithjustafew
additionaltrainingexamples,theresultdependsheavilyontheinputprompt[6]. Thisnecessitates
anempiricalartofcomposingandformattingtheprompttomaximizeamodel’sperformanceon
adesiredtask, whichisknownaspromptengineeringorprompthacking. Fine-tuningretrainsa
modelpre-trainedongeneraldomainstoaspecifictask[9,31]. Variantsofitincludelearningjusta
subsetoftheparameters[8,9],yetpractitionersoftenretrainallofthemtomaximizethedownstream
performance. However,theenormityofGPT-3175Bmakesitchallengingtoperformfine-tuningin
theusualwayduetothelargecheckpointitproducesandthehighhardwarebarriertoentrysinceit
hasthesamememoryfootprintaspre-training.
Parameter-EfficientAdaptation. [15,34]proposeinsertingadapterlayersbetweenexistinglayers
inaneuralnetwork. Ourmethodusesabottleneckstructuresimilarto[15]toimposealow-rank
constraintontheweightupdates. Thekeyfunctionaldifferenceisthatourlearnedweightscanbe
mergedwiththemainweightsduringinference,thusnotintroducinganylatency,whichisnotthe
case for the adapter layers. More recently, [14, 19, 21, 25] proposed optimizing the input word
embeddingsinlieuoffine-tuning,akintoacontinuousanddifferentiablegeneralizationofprompt
engineering. Weincludecomparisonswith[21]inourexperimentsection. However,thislineof
workscanonlyscaleupbyusingmorespecialtokensintheprompt,whichtakeupavailablesequence
lengthfortasktokenswhenpositionalembeddingsarelearned.
Low-RankStructuresinDeepLearning. Low-rankstructureisverycommoninmachinelearning.
Alotofmachinelearningproblemshavecertainintrinsiclow-rankstructure[7,13,23,24].Moreover,
itisknownthatformanydeeplearningtasks,especiallythosewithaheavilyover-parametrizedneural
network,thelearnedneuralnetworkwillenjoylow-rankpropertiesaftertraining[29]. Someprior
worksevenexplicitlyimposethelow-rankconstraintwhentrainingtheoriginalneuralnetwork,such
as[17,30,35,38,39],however,tothebestofourknowledge,noneoftheseworksconsiderslow-rank
updateforadaptationtodownstreamtasks. Intheoryliterature,itisknownthatneuralnetworks
outperformotherclassicallearningmethods,includingthecorresponding(finite-width)neuraltangent
kernels[5,22]whentheunderlyingconceptclasshascertainlow-rankstructure[2,3,11]. Another
3Westillneedthe350GBmodelduringdeployment;however,storing100adaptedmodelsonlyrequires
350GB+35MB*100≈354GBasopposedto100*350GB≈35TB.
4

<!-- page 5 -->

theoreticalresultin[4]suggeststhatlow-rankadaptationscanbeusefulforadversarialtraining. In
sum,webelievethatourproposedlow-rankadaptationupdateiswell-motivatedbytheliterature.
5 EmpiricalExperiments
We benchmark the downstream performance of LoRA on both GPT-2 and GPT-3. Specifically,
we focus on WikiSQL [40] (natural language to SQL queries), MultiNLI [37] (natural language
inference)4,andSAMSum[12](conversationsummarization)forGPT-3.Foradirectcomparison
with [21] on GPT-2, we follow their setup and use Table-to-Text E2E NLG Challenge [28], a
language generation task. See Appendix B for more details on these datasets. We also compare
withtheothertuningmethodssuchas[21,25]. WeuseNVIDIATeslaV100GPUsforallofour
experiments. ForGPT-3,weuse96GPUsforfine-tuningand24GPUsforLoRA.Weuseasingle
GPUforGPT-2experiments.
5.1 Baselines
Wecomparewiththefollowingbaselines. Notethatweimplementedbothprefix-embeddingand
prefix-layertuninginGPT-3onourownaccordingto[21],whilethenon-LoRAbaselinesallcome
fromtheimplementationof[21].
Fine-Tuningisthedefactodefaultadaptationapproach,wherethemodelisinitializedtothepre-
trainedparameters. Duringfine-tuning,allthemodelparametersundergogradientupdatesaccording
toEqn. 1. Asimplevariantofitistoupdateonlysomelayerswhilefreezingothers. Weincludeone
suchbaselinereportedinpriorwork[21],whichtunesjustthelasttwolayers(FT-Top2).
Biasonlyisabaselinewhereweonlytrainthebiasvectorswhenadaptingtodownstreamtasks.
Prefix-embeddingtuninginjectsspecialtokensalongsidetheinputtokens. Thesespecialtokens
havetrainablewordembeddingsandaregenerallynotinthemodel’svocabulary. Theplacementof
suchtokenscanhaveanimpactonperformance. Wefocuson“prefixing”,whichprependsthese
tokenstotheprompt,and“infixing”,whichappendstotheprompt;botharediscussedin[21]. We
usel (resp. l )denotethenumberofprefix(resp. infix)tokens. Thenumberoftrainableparameters
p i
is|Θ|=d ×(l +l ).
model p i
Prefix-layertuningisanextensiontoprefix-embeddingtuning. Insteadofjustlearningtheword
embeddings (or equivalently, the activations after the embedding layer) for some special tokens,
we learn the activations after every layer. The activations computed from previous layers are
simply replaced by our trainable ones. The resulting number of trainable parameters is |Θ| =
L×d ×(l +l ),whereListhenumberoflayers(96forGPT-3).
model p i
Adaptertuning[15]insertsadapterlayersbetweentheself-attentionmodule(andtheMLPmodule)
andthesubsequentresidualconnection. Therearetwoweightmatricesinanadapterlayerwitha
nonlinearityinbetween. Thedimensionofthehiddenlayerinanadapterlayerdeterminesthetotal
numberoftrainableparameters. Forahiddensizeofn,wehave|Θ|=4×L×d ×n.
model
LoRAaddstrainablepairsofrankdecompositionmatricestoexistingweightmatrices. Asmentioned
inSec.3.1andsupportedbyouranalysisinSec.6.1,weonlyapplyLoRAtoW andW inour
q v
experiments. Thenumberoftrainableparametersisdeterminedbytherankrandtheshapeofthe
originalweights: |Θ|=2×L×d ×r.
model
5.2 PerformanceonGPT-3
Hyperparameters ForallofourGPT-3experiments,wetrainusingAdamW[26]for2epochs
withabatchsizeof100ktokens,asequencelengthof768,andaweightdecayfactorof0.1. We
tune the learning rate for all method-dataset combinations. See Sec. C.1 for more details. For
prefix-embeddingtuning,wefindtheoptimall andl tobe256and8,respectively,totalling3.2M
p i
trainableparameters. Weusel =8andl =8inPrefixLayerwith20.2M trainableparametersto
p i
obtaintheoverallbestperformance. WepresenttwoconfigurationsforLoRA:onewithr =r =4
q v
4RecentworkstreatMultiNLI-matchedandMultiNLI-mismatchedasseparatetasks.Inthiswork,weonly
reportaccuracyonMultiNLI-matchedvalidationset.
5

<!-- page 6 -->

#ofTrainable WikiSQL MNLI-m SAMSum
Method
Parameters Accuracy(%) Accuracy(%) R1/R2/RL
GPT-3175B(Fine-Tune) 175,255.8M 73.0 89.5 52.0/28.0/44.5
GPT-3175B(BiasOnly) 14.2M 71.3 91.0 51.3/27.4/43.5
GPT-3175B(PrefixEmbed) 3.2M 63.1 88.6 48.3/24.2/40.5
GPT-3175B(PrefixLayer) 20.2M 70.1 89.5 50.8/27.3/43.5
GPT-3175B(LoRA) 4.7M 73.4 91.3 52.1/28.3/44.3
GPT-3175B(LoRA) 37.7M 73.8 91.7 53.2/29.2/45.0
Table1: LogicalformvalidationaccuracyonWikiSQL,validationaccuracyonMultiNLI-matched
andRouge-1/2/LonSAMSumachievedbydifferentGPT-3adaptationmethods. LoRAperforms
better than prior approaches, including conventional fine-tuning. The result on WikiSQL has a
fluctuationof±0.3%andMNLI-m±0.1%.
(18.8M)andonewithr =r =8(37.7M).
q v
AsshowninTable1,onallthreedatasets,LoRAoutperformsthefine-tuningbaseline.Itmightappear
thatprefix-embeddingtuningcouldbenefitfromhavingmoretrainableparameterssinceitusesmuch
fewerparametersthanothermethods. Nonetheless,thisisnotthecaseasshowninFig.2. Weobserve
asignificantperformancedropwhenweusemorethan256specialtokensforprefix-embeddingtuning
ormorethan32specialtokensforprefix-layertuning. Thiscorroboratesthesimilarobservation
reportedin[21]. Whileathoroughinvestigationintothisphenomenonisout-of-scopeforthiswork,
wesuspectthathavingmorespecialtokenscausestheinputdistributiontoshiftfurtherawayfrom
thepre-trainingdatadistribution,degradingthemodelperformance. Separately,weinvestigatethe
performanceofadaptationapproachesinthelow-dataregimeinSec.E.3.
0.75
0.70
0.65
0.60
0.55
6 7 8 9 10 11
log10 # Trainable Parameters
ycaruccA
noitadilaV
WikiSQL MultiNLI-matched
0.92
0.90
Method 0.88
Fine-Tune
PrefixEmbed 0.86
PrefixLayer
LoRA 0.84
6 7 8 9 10 11
log10 # Trainable Parameters
Figure2: GPT-3175Bvalidationaccuracyvs. numberoftrainableparametersofseveraladaptation
methodsonWikiSQLandMNLI-matched. LoRAenjoysbetterscalabilityandtaskperformance.
SeeSec.E.1formoredetailsontheplotteddatapoints.
5.3 PerformanceonGPT-2
HavingestablishedLoRAasacompetitivemethodonGPT-3,wehopetoanswerifLoRAcanalso
workonlessoverlyparametrizedmodels,suchasGPT-2mediumorlarge[32]. Wekeepoursetupas
closeaspossibleto[21]foradirectcomparison. Duetospaceconstraint,weonlypresentourresult
onE2ENLGChallenge(Table2)inthissection–seeSec.E.2forresultsonmoredatasets.
Hyperparameters WetrainallofourGPT-2modelsusingAdamW[26]andalinearlearningrate
schedulefor5epochs. Weusethebatchsize,learningrate,andbeamsearchbeamsizedescribed
in[21]. Accordingly,wealsotunethesehyperparametersforLoRA,andweincludealistoftheused
hyperparametersinTable8.
6

<!-- page 7 -->

Method #ofTrainable E2E
Parameters BLEU NIST MET ROUGE-L CIDEr
GPT-2M(Fine-Tune) 354.92M 68.2 8.62 46.2 71.0 2.47
GPT-2M(Adapter) 11.48M 68.9 8.71 46.1 71.3 2.47
GPT-2M(FT-Top2) 25.19M 68.1 8.59 46.0 70.8 2.41
GPT-2M(Prefix) 0.35M 69.7 8.81 46.1 71.4 2.49
GPT-2M(LoRA) 0.35M 70.4 8.85 46.8 71.8 2.53
GPT-2L(Fine-Tune) 774.03M 68.5 8.78 46.0 69.9 2.45
GPT-2L(Prefix) 0.77M 70.3 8.85 46.2 71.7 2.47
GPT-2L(LoRA) 0.77M 70.4 8.89 46.8 72.0 2.47
Table 2: GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG
Challenge. Forallmetrics,higherisbetter. LoRAoutperformsseveralbaselineswithcomparableor
fewertrainableparameters.
6 UnderstandingtheLow-RankUpdates
GiventheempiricaladvantageofLoRA,wehopetofurtherexplainthepropertiesofthelow-rank
adaptation learned from downstream tasks. Note that the low-rank structure not only lowers the
hardwarebarriertoentrywhichallowsustorunmultipleexperimentsinparallel,butalsogivesbetter
interpretabilityofhowtheupdateweightsarecorrelatedwiththepre-trainedweights.
Weperformasequenceofempiricalstudiestoanswerthefollowingquestions: 1)Givenaparameter
budgetconstraint,whichsubsetofweightmatricesinapre-trainedTransformershouldweadaptto
maximizedownstreamperformance?2)Isthe“optimal”adaptationmatrix∆W reallyrank-deficient?
Ifso,whatisagoodranktouseinpractice? 3)Whatistheconnectionbetween∆W andW? Does
∆W highlycorrelatewithW? Howlargeis∆W comparingtoW?
Webelievethatouranswerstoquestion(2)and(3)shedlightonthefundamentalprinciplesofusing
pre-trainedlanguagemodelsfordownstreamtasks,whichisacriticaltopicinNLP.
6.1 WhichWeightMatricesinTransformerShouldWeApplyLoRAto?
Givenalimitedparameterbudget,whichtypesofweightsshouldweadaptwithLoRAtoobtainthe
bestperformanceondownstreamtasks? AsmentionedinSec.3.1,weonlyconsiderweightmatrices
intheself-attentionmodule. Wesetaparameterbudgetof18M(roughly35MB)onGPT-3,which
correspondstor =8ifweadaptonetypeofweightsorr =4ifweadapttwotypes,forall96layers.
TheresultispresentedinTable3.
#ofTrainableParameters=18M
WeightType W W W W W ,W W ,W
q k v o q k q v
Rankr 8 8 8 8 4 4
WikiSQL(±0.3%) 70.4 70.0 73.0 73.2 71.4 73.7
MultiNLI(±0.1%) 91.0 90.8 91.0 91.3 91.3 91.3
Table3: ValidationaccuracyonWikiSQLandMultiNLIafterapplyingLoRAtodifferenttypesof
attentionweightsinGPT-3,giventhesamenumberoftrainableparameters. AdaptingbothW and
q
W givesthebestperformanceoverall. Wefindthestandarddeviationacrossrandomseedstobe
v
consistentforagivendataset,whichwereportinthefirstcolumn.
Note that putting all the parameters in ∆W or ∆W results in significantly lower performance,
q k
whileadaptingbothW andW yieldsthebestresult. Thissuggeststhatevenarankoffourcaptures
q v
enoughinformationin∆W suchthatitispreferabletoadaptmoreweightmatricesthanadaptinga
singletypeofweightswithalargerrank.
7

<!-- page 8 -->

6.2 WhatistheOptimalRankrforLoRA?
Weturnourattentiontotheeffectofrankronmodelperformance. WeadaptbothW andW since
q v
theyperformedthebestinthepreviousexperimentandaswellasW aloneforcomparison.
q
WeightType r =1 r =2 r =4 r =8 r =64
W ,W 73.4 73.3 73.7 73.8 73.5
WikiSQL(±0.3%) q v
W 68.8 69.6 70.5 70.4 70.0
q
MultiNLI(±0.1%) W ,W 91.3 91.4 91.3 91.7 91.4
q v
Table4: ValidationaccuracyonWikiSQLandMultiNLIwithdifferentrankr. Tooursurprise,a
rankassmallasonesufficesforadaptingbothW andW onthesedatasetswhiletrainingW alone
q v q
needsalargerr. WereplicatethisonGPT-2inSec.G.2.
Table4showsthat,surprisingly,LoRAalreadyperformscompetitivelywithaverysmallr(moreso
forW thanW ). Thissuggeststheupdatematrix∆W couldhaveaverysmall“intrinsicrank".5
q+v q
Tofurthersupportthisfinding,wechecktheoverlapofthesubspaceslearnedbydifferentchoices
ofr andbydifferentrandomseeds. Wearguethatincreasingr doesnotcovermoremeaningful
subspaces,whichsuggeststhatalow-rankadaptationmatrixissufficient.
Subspacesimilaritybetweendifferentr. GivenA andA whicharethelearnedadaptation
r=8 r=64
matrices with rank r = 8 and 64 using the same pre-trained model, we perform singular value
decompositionandobtaintheright-singularunitarymatricesU andU .6 Wehopetoanswer:
Ar=8 Ar=64
howmuchofthesubspacespannedbythetopisingularvectorsinU (for1≤i≤8)iscontained
Ar=8
inthesubspacespannedbytopj singularvectorsofU (for1 ≤ j ≤ 64)? Wemeasurethis
Ar=64
quantitywithanormalizedsubspacesimilaritybasedontheGrassmanndistance(SeeAppendixF
foramoreformaldiscussion)
||Ui(cid:62) Uj ||2
φ(A ,A ,i,j)= Ar=8 Ar=64 F ∈[0,1] (4)
r=8 r=64 min(i,j)
whereUi representsthecolumnsofU correspondingtothetop-isingularvectors.
Ar=8 Ar=8
φ(·) has a range of [0,1], where 1 represents a complete overlap of subspaces and 0 a complete
separation. SeeFig.3forhowφchangesaswevaryiandj. Weonlylookatthe48thlayer(outof
96)duetospaceconstraint,buttheconclusionholdsforotherlayersaswell,asshowninSec.G.1.
1.0
0.8
0.6
0.4
0.2
0.0
1 6 21 81 32 92 53 04 64 25 85
j
i
1
2
3
4
5
6
7
8
Wq
1 6 21 81 32 92 53 04 64 25 85
(Ar=64,Ar=8,i,j)
Wv Wq Wv
1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
j j j
Figure3: SubspacesimilaritybetweencolumnvectorsofA andA forboth∆W and∆W .
r=8 r=64 q v
Thethirdandthefourthfigureszoominonthelower-lefttriangleinthefirsttwofigures. Thetop
directionsinr =8areincludedinr =64,andviceversa.
5However,wedonotexpectasmallrtoworkforeverytaskordataset. Considerthefollowingthought
experiment:ifthedownstreamtaskwereinadifferentlanguagethantheoneusedforpre-training,retrainingthe
entiremodel(similartoLoRAwithr=d )couldcertainlyoutperformLoRAwithasmallr.
model
6NotethatasimilaranalysiscanbecarriedoutwithBandtheleft-singularunitarymatrices–westickwith
Aforourexperiments.
8

<!-- page 9 -->

WemakeanimportantobservationfromFig.3.
DirectionscorrespondingtothetopsingularvectoroverlapsignificantlybetweenA
r=8
andA ,whileothersdonot. Specifically,∆W (resp. ∆W )ofA and∆W
r=64 v q r=8 v
(resp. ∆W )ofA shareasubspaceofdimension1withnormalizedsimilarity
q r=64
>0.5,providinganexplanationofwhyr =1performsquitewellinourdownstream
tasksforGPT-3.
SincebothA andA arelearnedusingthesamepre-trainedmodel,Fig.3indicatesthatthetop
r=8 r=64
singular-vectordirectionsofA andA arethemostuseful,whileotherdirectionspotentially
r=8 r=64
containmostlyrandomnoisesaccumulatedduringtraining. Hence,theadaptationmatrixcanindeed
haveaverylowrank.
0.5
0.4
0.3
0.2
0.1
0.0
1 5 01 51 02 52 03 43 93 44 94 45 95
1
8
16
24
32
40
48
56
j
i
Wq
1 5 01 51 02 52 03 43 93 44 94 45 95
(Ar=64,A0r=64,i,j)
Wv
j
1 5 01 51 02 52 03 43 93 44 94 45 95
Random Gaussian
j
Figure4: LeftandMiddle: NormalizedsubspacesimilaritybetweenthecolumnvectorsofA
r=64
from two random seeds, for both ∆W and ∆W in the 48-th layer. Right: the same heat-map
q v
betweenthecolumnvectorsoftworandomGaussianmatrices. SeeSec.G.1forotherlayers.
Subspace similarity between different random seeds. We further confirm this by plotting the
normalizedsubspacesimilaritybetweentworandomlyseededrunswithr = 64,showninFig.4.
∆W appears to have a higher “intrinsic rank” than ∆W , since more common singular value
q v
directions are learned by both runs for ∆W , which is in line with our empirical observation
q
inTable4. Asacomparison,wealsoplottworandomGaussianmatrices,whichdonotshareany
commonsingularvaluedirectionswitheachother.
6.3 HowDoestheAdaptationMatrix∆W ComparetoW?
Wefurtherinvestigatetherelationshipbetween∆W andW. Inparticular,does∆W highlycorrelate
withW? (Ormathematically,is∆W mostlycontainedinthetopsingulardirectionsofW?) Also,
how“large”is∆W comparingtoitscorrespondingdirectionsinW? Thiscanshedlightonthe
underlyingmechanismforadaptingpre-trainedlanguagemodels.
Toanswerthesequestions,weprojectW ontother-dimensionalsubspaceof∆W bycomputing
U(cid:62)WV(cid:62),withU/V beingtheleft/rightsingular-vectormatrixof∆W.Then,wecomparetheFrobe-
niusnormbetween(cid:107)U(cid:62)WV(cid:62)(cid:107) and(cid:107)W(cid:107) . Asacomparison,wealsocompute(cid:107)U(cid:62)WV(cid:62)(cid:107) by
F F F
replacingU,V withthetoprsingularvectorsofW orarandommatrix.
r =4 r =64
∆W W Random ∆W W Random
q q q q
||U(cid:62)W V(cid:62)|| = 0.32 21.67 0.02 1.90 37.71 0.33
q F
||W || =61.95 ||∆W || =6.91 ||∆W || =3.57
q F q F q F
Table5: TheFrobeniusnormofU(cid:62)W V(cid:62) whereU andV aretheleft/righttoprsingularvector
q
directionsofeither(1)∆W ,(2)W ,or(3)arandommatrix. Theweightmatricesaretakenfrom
q q
the48thlayerofGPT-3.
9

<!-- page 10 -->

WedrawseveralconclusionsfromTable5. First,∆W hasastrongercorrelationwithW comparedto
arandommatrix,indicatingthat∆W amplifiessomefeaturesthatarealreadyinW. Second,instead
ofrepeatingthetopsingulardirectionsofW,∆W onlyamplifiesdirectionsthatarenotemphasized
inW. Third,theamplificationfactorisratherhuge: 21.5≈6.91/0.32forr =4. SeeSec.G.4for
whyr =64hasasmalleramplificationfactor. WealsoprovideavisualizationinSec.G.3forhow
thecorrelationchangesasweincludemoretopsingulardirectionsfromW . Thissuggeststhatthe
q
low-rankadaptationmatrixpotentiallyamplifiestheimportantfeaturesforspecificdownstreamtasks
thatwerelearnedbutnotemphasizedinthegeneralpre-trainingmodel.
7 ConclusionandFutureWork
Fine-tuning enormous language models is prohibitively expensive in terms of both the hardware
requirementandthestorage/switchingcostforhostingmultipleinstances. WeproposeLoRA,an
efficient adaptation strategy that neither introduces inference latency nor reduces input sequence
lengthwhileretainingmodelquality. Importantly,itallowsforquicktask-switchingwhendeployed
asaservicebysharingthevastmajorityofthemodelparameters. WhilewefocusedonTransformer,
theproposedprinciplesaregenerallyapplicabletoanyneuralnetworkswithdenselayers.
LoRAcanpotentiallyworkintandemwithotherfine-tuningtechniques. Inthefuture,wehopeto
exploreonlytuningsomelayersoraddingadversarialtraining. Finally,therank-deficiencyof∆W
suggeststhatW couldberank-deficientaswell,whichmightinspirelotsoffutureworks.
Acknowledgments
WethankinalphabeticalorderJianfengGao,JadeHuang,JiayuanHuang,LisaXiangLi,Xiaodong
Liu,YabinLiu,BenjaminVanDurme,LuisVargas,HaoranWei,PeterWelinder,andGregYangfor
providingvaluablefeedback.
References
[1] ArmenAghajanyan,LukeZettlemoyer,andSonalGupta. IntrinsicDimensionalityExplainsthe
EffectivenessofLanguageModelFine-Tuning. arXiv:2012.13255[cs],December2020. URL
http://arxiv.org/abs/2012.13255.
[2] ZeyuanAllen-ZhuandYuanzhiLi. WhatCanResNetLearnEfficiently,GoingBeyondKernels?
InNeurIPS,2019. Fullversionavailableathttp://arxiv.org/abs/1905.10337.
[3] ZeyuanAllen-ZhuandYuanzhiLi. Backwardfeaturecorrection: Howdeeplearningperforms
deeplearning. arXivpreprintarXiv:2001.04413,2020.
[4] ZeyuanAllen-ZhuandYuanzhiLi. Featurepurification: Howadversarialtrainingperforms
robustdeeplearning. arXivpreprintarXiv:2005.10190,2020.
[5] ZeyuanAllen-Zhu,YuanzhiLi,andZhaoSong. Aconvergencetheoryfordeeplearningvia
over-parameterization. InICML,2019. Fullversionavailableathttp://arxiv.org/abs/
1811.03962.
[6] TomB.Brown,BenjaminMann,NickRyder,MelanieSubbiah,JaredKaplan,PrafullaDhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agar-
wal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh,
DanielM.Ziegler,JeffreyWu,ClemensWinter,ChristopherHesse,MarkChen,EricSigler,
MateuszLitwin,ScottGray,BenjaminChess,JackClark,ChristopherBerner,SamMcCandlish,
AlecRadford,IlyaSutskever,andDarioAmodei. LanguageModelsareFew-ShotLearners.
arXiv:2005.14165[cs],July2020. URLhttp://arxiv.org/abs/2005.14165.
[7] Jian-FengCai,EmmanuelJCandès,andZuoweiShen. Asingularvaluethresholdingalgorithm
formatrixcompletion. SIAMJournalonoptimization,20(4):1956–1982,2010.
[8] RonanCollobertandJasonWeston.Aunifiedarchitecturefornaturallanguageprocessing:deep
neuralnetworkswithmultitasklearning. InProceedingsofthe25thinternationalconference
10

<!-- page 11 -->

onMachinelearning,ICML’08,pages160–167,NewYork,NY,USA,July2008.Association
forComputingMachinery. ISBN978-1-60558-205-4. doi: 10.1145/1390156.1390177. URL
https://doi.org/10.1145/1390156.1390177.
[9] JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova. BERT:Pre-trainingof
DeepBidirectionalTransformersforLanguageUnderstanding. arXiv:1810.04805[cs],May
2019. URLhttp://arxiv.org/abs/1810.04805. arXiv: 1810.04805.
[10] ClaireGardent,AnastasiaShimorina,ShashiNarayan,andLauraPerez-Beltrachini.Thewebnlg
challenge: Generatingtextfromrdfdata. InProceedingsofthe10thInternationalConference
onNaturalLanguageGeneration,pages124–133,2017.
[11] BehroozGhorbani,SongMei,TheodorMisiakiewicz,andAndreaMontanari. Whendoneural
networksoutperformkernelmethods? arXivpreprintarXiv:2006.13409,2020.
[12] BogdanGliwa,IwonaMochol,MaciejBiesek,andAleksanderWawer. Samsumcorpus: A
human-annotateddialoguedatasetforabstractivesummarization. CoRR,abs/1911.12237,2019.
URLhttp://arxiv.org/abs/1911.12237.
[13] LarsGrasedyck,DanielKressner,andChristineTobler. Aliteraturesurveyoflow-ranktensor
approximationtechniques. GAMM-Mitteilungen,36(1):53–78,2013.
[14] KarenHambardzumyan,HrantKhachatrian,andJonathanMay. WARP:Word-levelAdversarial
ReProgramming. arXiv:2101.00121[cs],December2020. URLhttp://arxiv.org/abs/
2101.00121. arXiv: 2101.00121.
[15] NeilHoulsby,AndreiGiurgiu,StanislawJastrzebski,BrunaMorrone,QuentindeLaroussilhe,
AndreaGesmundo,MonaAttariyan,andSylvainGelly. Parameter-EfficientTransferLearning
for NLP. arXiv:1902.00751 [cs, stat], June 2019. URL http://arxiv.org/abs/1902.
00751.
[16] Yen-ChangHsu,Yen-ChengLiu,AnitaRamasamy,andZsoltKira. Re-evaluatingContinual
LearningScenarios: ACategorizationandCaseforStrongBaselines. arXiv:1810.12488[cs],
January2019. URLhttp://arxiv.org/abs/1810.12488. arXiv: 1810.12488.
[17] MaxJaderberg,AndreaVedaldi,andAndrewZisserman. Speedingupconvolutionalneural
networkswithlowrankexpansions. arXivpreprintarXiv:1405.3866,2014.
[18] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins,
AndreiA.Rusu,KieranMilan,JohnQuan,TiagoRamalho,AgnieszkaGrabska-Barwinska,
DemisHassabis,ClaudiaClopath,DharshanKumaran,andRaiaHadsell. Overcomingcatas-
trophic forgetting in neural networks. arXiv:1612.00796 [cs, stat], January 2017. URL
http://arxiv.org/abs/1612.00796. arXiv: 1612.00796.
[19] BrianLester,RamiAl-Rfou,andNoahConstant. ThePowerofScaleforParameter-Efficient
PromptTuning. arXiv:2104.08691[cs],April2021. URLhttp://arxiv.org/abs/2104.
08691. arXiv: 2104.08691.
[20] Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the Intrinsic
DimensionofObjectiveLandscapes. arXiv:1804.08838[cs, stat], April2018. URLhttp:
//arxiv.org/abs/1804.08838. arXiv: 1804.08838.
[21] XiangLisaLiandPercyLiang. Prefix-Tuning: OptimizingContinuousPromptsforGeneration.
arXiv:2101.00190[cs],January2021. URLhttp://arxiv.org/abs/2101.00190.
[22] Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic
gradientdescentonstructureddata. InAdvancesinNeuralInformationProcessingSystems,
2018.
[23] YuanzhiLi, YingyuLiang, andAndrejRisteski. Recoveryguaranteeofweightedlow-rank
approximationviaalternatingminimization. InInternationalConferenceonMachineLearning,
pages2358–2367.PMLR,2016.
11

<!-- page 12 -->

[24] YuanzhiLi,TengyuMa,andHongyangZhang.Algorithmicregularizationinover-parameterized
matrixsensingandneuralnetworkswithquadraticactivations. InConferenceOnLearning
Theory,pages2–47.PMLR,2018.
[25] XiaoLiu, YananZheng, ZhengxiaoDu, MingDing, YujieQian, ZhilinYang, andJieTang.
GPTUnderstands,Too. arXiv:2103.10385[cs],March2021. URLhttp://arxiv.org/abs/
2103.10385. arXiv: 2103.10385.
[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101,2017.
[27] LinyongNan,DragomirRadev,RuiZhang,AmritRau,AbhinandSivaprasad,ChiachunHsieh,
XiangruTang,AaditVyas,NehaVerma,PranavKrishna,etal. Dart: Open-domainstructured
datarecordtotextgeneration. arXivpreprintarXiv:2007.02871,2020.
[28] JekaterinaNovikova,OndˇrejDušek,andVerenaRieser. Thee2edataset: Newchallengesfor
end-to-endgeneration. arXivpreprintarXiv:1706.09254,2017.
[29] SametOymak,ZalanFabian,MingchenLi,andMahdiSoltanolkotabi. Generalizationguaran-
teesforneuralnetworksviaharnessingthelow-rankstructureofthejacobian. arXivpreprint
arXiv:1906.05392,2019.
[30] DanielPovey,GaofengCheng,YimingWang,KeLi,HainanXu,MahsaYarmohammadi,and
SanjeevKhudanpur. Semi-orthogonallow-rankmatrixfactorizationfordeepneuralnetworks.
InInterspeech,pages3743–3747,2018.
[31] AlecRadford,KarthikNarasimhan,TimSalimans,andIlyaSutskever. ImprovingLanguage
UnderstandingbyGenerativePre-Training. page12,.
[32] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
LanguageModelsareUnsupervisedMultitaskLearners. page24,.
[33] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
Languagemodelsareunsupervisedmultitasklearners. OpenAIblog,1(8):9,2019.
[34] Sylvestre-AlviseRebuffi,HakanBilen,andAndreaVedaldi. Learningmultiplevisualdomains
withresidualadapters. arXiv:1705.08045[cs,stat],November2017. URLhttp://arxiv.
org/abs/1705.08045. arXiv: 1705.08045.
[35] TaraNSainath,BrianKingsbury,VikasSindhwani,EbruArisoy,andBhuvanaRamabhadran.
Low-rankmatrixfactorizationfordeepneuralnetworktrainingwithhigh-dimensionaloutput
targets. In2013IEEEinternationalconferenceonacoustics,speechandsignalprocessing,
pages6655–6659.IEEE,2013.
[36] AshishVaswani,NoamShazeer,NikiParmar,JakobUszkoreit,LlionJones,AidanNGomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st
InternationalConferenceonNeuralInformationProcessingSystems,pages6000–6010,2017.
[37] AdinaWilliams, NikitaNangia, andSamuelBowman. A broad-coveragechallenge corpus
forsentenceunderstandingthroughinference. InProceedingsofthe2018Conferenceofthe
NorthAmericanChapteroftheAssociationforComputationalLinguistics: HumanLanguage
Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana, June
2018.AssociationforComputationalLinguistics. doi: 10.18653/v1/N18-1101. URLhttps:
//www.aclweb.org/anthology/N18-1101.
[38] YuZhang,EkapolChuangsuwanich,andJamesGlass. Extractingdeepneuralnetworkbottle-
neckfeaturesusinglow-rankmatrixfactorization. In2014IEEEinternationalconferenceon
acoustics,speechandsignalprocessing(ICASSP),pages185–189.IEEE,2014.
[39] Yong Zhao, Jinyu Li, and Yifan Gong. Low-rank plus diagonal adaptation for deep neural
networks. In2016IEEEInternationalConferenceonAcoustics,SpeechandSignalProcessing
(ICASSP),pages5005–5009.IEEE,2016.
[40] VictorZhong,CaimingXiong,andRichardSocher. Seq2sql: Generatingstructuredqueries
from natural language using reinforcement learning. CoRR, abs/1709.00103, 2017. URL
http://arxiv.org/abs/1709.00103.
12

<!-- page 13 -->

A LargeLanguageModelsStillNeedParameterUpdates
Few-shotlearning,orpromptengineering,isveryadvantageouswhenweonlyhaveahandfulof
trainingsamples. However,inpractice,wecanoftenaffordtocurateafewthousandormoretraining
examplesforperformance-sensitiveapplications. AsshowninTable6, fine-tuningimprovesthe
model performance drastically compared to few-shot learning on datasets large and small. We
taketheGPT-3few-shotresultonRTEfromtheGPT-3paper[6]. ForMNLI-mtask,weusetwo
demonstrationsperclassorsixin-contextexamplesintotal.
Method MNLI-m(Val. Acc./%) RTE(Val. Acc./%)
GPT-3Few-Shot 40.6 69.0
GPT-3Fine-Tuned 89.5 85.4
Table6: Fine-tuningsignificantlyoutperformsfew-shotlearningonGPT-3[6].
B DatasetDetails
MultiNLI is a natural language inference dataset [37], consisting of 392,702 training and
9,815 validation examples. For pre-processing, we encode the context x = {[premise] :
,premise,[hypothesis] :,hypothesis}; target y ∈ {entailment,neutral,contradiction}. The
datasetismostlyreleasedundertheOANC’slicense,andsubsetsofitareunderotherpermissive
licensessuchastheCreativeCommonsShare-Alike3.0UnportedLicense.
WikiSQLisintroducedin [40]andcontains56,355/8,421training/validationexamples. Thetaskis
togenerateSQLqueriesfromnaturallanguagequestionsandtableschemata. Weencodecontextas
x={tableschema,query}andtargetasy ={SQL}. ThedatasetisreleaseundertheBSD3-Clause
License.
E2ENLGChallengewasfirstintroducedin [28]asadatasetfortrainingend-to-end,data-driven
naturallanguagegenerationsystemsandiscommonlyusedfordata-to-textevaluation. TheE2E
datasetconsistsof 42K training, 4.6K validation, and 4.6K testexamplesfromtherestaurant
domain. Each source table used as inputcan have multiple references. Each sample input (x,y)
consistsofasequenceofslot-valuepairs,alongwithacorrespondingnaturallanguagereferencetext.
ThedatasetisreleasedunderCreativeCommonsBY-NC-SA4.0.
DART is an open-domain data-to-text dataset described in [27]. DART inputs are structured as
sequencesofENTITY|RELATION|ENTITYtriples. With 82K examplesintotal,DARTisa
significantlylargerandmorecomplexdata-to-texttaskcomparedtoE2E.Thedatasetisreleased
undertheMITlicense.
WebNLGisanothercommonlyuseddatasetfordata-to-textevaluation[10]. With 22K examplesin
totalWebNLGcomprises14distinctcategories,nineofwhichareseenduringtraining. Sincefive
ofthe14totalcategoriesarenotseenduringtraining,butarerepresentedinthetestset,evaluation
istypicallybrokenoutby“seen”categories(S),“unseen”categories(U)and“all”(A).Eachinput
exampleisrepresentedbyasequenceofSUBJECT|PROPERTY|OBJECTtriples. Thedatasetis
releasedunderCreativeCommonsBY-NC-SA4.0.
C HyperparametersUsedinExperiments
C.1 GPT-3Experiments
ThetraininghyperparametersusedinourGPT-3experimentsarelistedinTable7.
C.2 GPT-2LoRA
ThehyperparametersusedforLoRAinGPT-2arelistedinTable8. Forthoseusedforotherbaselines,
see[21].
13

<!-- page 14 -->

Hyperparameters Fine-Tune Prefix-Embed Prefix-Tune LoRA
Optimizer AdamW
BatchSize 128
#Epoch 2
WarmupTokens 250,000
LearningRateSchedule Linear
LearningRate 5.00E-06 5.00E-04 1.00E-04 2.00E-04
Table7: ThetraininghyperparametersusedfordifferentGPT-3adaptionmethods. Weusethesame
hyperparametersforalldatasets.
Dataset E2E WebNLG DART
Training
Optimizer AdamW
WeightDecay 0.01 0.01 0.0
DropoutProb 0.1 0.1 0.0
BatchSize 8
#Epoch 5
WarmupSteps 500
LearningRateSchedule Linear
LabelSmooth 0.1 0.1 0.0
LearningRate 0.0002
Adaptation r =r =4
q v
LoRAα 32
Inference
BeamSize 10
LengthPenalty 0.9 0.8 0.8
norepeatngramsize 4
Table8: ThehyperparametersforGPT-2LoRAonE2E,WebNLGandDART.
D CombiningLoRAwithPrefixTuning
LoRAcanbenaturallycombinedwithexistingprefix-basedapproaches. Inthissection,weevaluate
twocombinationsofLoRAandvariantsofprefix-tuningonWikiSQLandMNLI.
LoRA+PrefixEmbed(LoRA+PE)combinesLoRAwithprefix-embeddingtuning,whereweinsert
l +l specialtokenswhoseembeddingsaretreatedastrainableparameters. Formoreonprefix-
p i
embeddingtuning,seeSec.5.1.
LoRA+PrefixLayer(LoRA+PL)combinesLoRAwithprefix-layertuning. Wealsoinsertl +l
p i
specialtokens;however,insteadoflettingthehiddenrepresentationsofthesetokensevolvenaturally,
we replace them after every Transformer block with an input agnostic vector. Thus, both the
embeddingsandsubsequentTransformerblockactivationsaretreatedastrainableparameters. For
moreonprefix-layertuning,seeSec.5.1.
InTable9,weshowtheevaluationresultsofLoRA+PEandLoRA+PLonWikiSQLandMultiNLI.
Firstofall,LoRA+PEsignificantlyoutperformsbothLoRAandprefix-embeddingtuningonWik-
iSQL,whichindicatesthatLoRAissomewhatorthogonaltoprefix-embeddingtuning. OnMultiNLI,
thecombinationofLoRA+PEdoesn’tperformbetterthanLoRA,possiblybecauseLoRAonitsown
alreadyachievesperformancecomparabletothehumanbaseline. Secondly,wenoticethatLoRA+PL
performsslightlyworsethanLoRAevenwithmoretrainableparameters. Weattributethistothefact
thatprefix-layertuningisverysensitivetothechoiceoflearningrateandthusmakestheoptimization
ofLoRAweightsmoredifficultinLoRA+PL.
14

<!-- page 15 -->

E AdditionalTask-BasedExperiments
E.1 AdditionalExperimentsonGPT-3
WepresentadditionalrunsonGPT-3withdifferentadaptationmethodsinTable9. Thefocusison
identifyingthetrade-offbetweenperformanceandthenumberoftrainableparameters.
Method Hyperparameters #TrainableParameters WikiSQL MNLI-m
Fine-Tune - 175B 73.0 89.5
l =32,l =8 0.39M 55.9 84.9
p i
l =64,l =8 0.88M 58.7 88.1
p i
PrefixEmbed l =128,l =8 1.67M 60.6 88.0
p i
l =256,l =8 3.24M 63.1 88.6
p i
l =512,l =8 6.40M 55.9 85.8
p i
l =2,l =2 5.06M 68.5 89.2
p i
l =8,l =0 10.1M 69.8 88.2
p i
PrefixLayer l =8,l =8 20.2M 70.1 89.5
p i
l =32,l =4 44.1M 66.4 89.6
p i
l =64,l =0 76.1M 64.9 87.9
p i
r =2 4.7M 73.4 91.7
v
r =r =1 4.7M 73.4 91.3
q v
LoRA r =r =2 9.4M 73.3 91.4
q v
r =r =4 18.8M 73.7 91.3
q v
r =r =8 37.7M 73.8 91.7
q v
r =r =64 301.9M 73.6 91.4
q v
r =r =8,l =8,l =4 37.8M 75.0 91.4
q v p i
LoRA+PE r =r =32,l =8,l =4 151.1M 75.9 91.1
q v p i
r =r =64,l =8,l =4 302.1M 76.2 91.3
q v p i
LoRA+PL r =r =8,l =8,l =4 52.8M 72.9 90.2
q v p i
Table9: HyperparameteranalysisofdifferentadaptationapproachesonWikiSQLandMNLI.Both
prefix-embeddingtuning(PrefixEmbed)andprefix-layertuning(PrefixLayer)performworseaswe
increasethenumberoftrainableparameters,whileLoRA’sperformancestabilizes. Performanceis
measuredinvalidationaccuracy.
E.2 AdditionalExperimentsonGPT-2
WealsorepeatourexperimentonDART[27]andWebNLG[10]followingthesetupof[21]. The
resultisshowninTable10. SimilartoourresultonE2ENLGChallenge,reportedinSec.5,LoRA
performs better than or at least on-par with prefix-based approaches given the same number of
trainableparameters.
E.3 Low-DataRegime
Toevaluatetheperformanceofdifferentadaptationapproachesinthelow-dataregime. werandomly
sample100,1kand10ktrainingexamplesfromthefulltrainingsetofMNLItoformthelow-data
MNLI-ntasks. InTable12,weshowtheperformanceofdifferentadaptationapproachesonMNLI-n.
To our surprise, PrefixEmbed and PrefixLayer performs very poorly on MNLI-100 dataset, with
PrefixEmbedperformingonlyslightlybetterthanrandomchance(37.6%vs. 33.3%). PrefixLayer
performsbetterthanPrefixEmbedbutisstillsignificantlyworsethanFine-TuneorLoRAonMNLI-
100. The gap between prefix-based approaches and LoRA/Fine-tuning becomes smaller as we
increasethenumberoftrainingexamples,whichmightsuggestthatprefix-basedapproachesarenot
suitableforlow-datatasksinGPT-3. LoRAachievesbetterperformancethanfine-tuningonboth
MNLI-100andMNLI-Full,andcomparableresultsonMNLI-1kandMNLI-10Kconsideringthe
15

<!-- page 16 -->

Method #Trainable DART
Parameters BLEU↑ MET↑ TER↓
GPT-2Medium
Fine-Tune 354M 46.0(±0.1) 0.39 0.46
Adapter 10M 45.4(±0.1) 0.38 0.46
FT-Top2 24M 38.1(±0.3) 0.34 0.56
Prefix 0.35M 45.7(±0.2) 0.38 0.46
LoRA 0.35M 47.1(±0.2) 0.39 0.46
GPT-2Large
Fine-Tune 774M 46.5(±0.1) 0.39 0.45
Prefix 0.77M 46.5(±0.2) 0.38 0.45
LoRA 0.77M 47.5(±0.1) 0.39 0.45
Table10: GPT-2withdifferentadaptationmethodsonDART.ThevariancesofMETandTERare
around0.01foralldifferentadaptionapproaches.
Method WebNLG
BLEU↑ MET↑ TER↓
U S A U S A U S A
GPT-2Medium
Fine-Tune(354M) 30.4(±.5) 63.2(±.3) 47.6(±.4) .32 .45 .39 .69 .34 .50
Adapter(10M) 47.9(±.2) 61.1(±.4) 55.2(±.3) .38 .43 .41 .45 .35 .39
FT-Top2(24M) 13.7(±.6) 50.1(±.4) 33.5(±.4) .16 .35 .26 1.03 .52 .75
Prefix(0.35M) 44.1(±.2) 63.1(±.1) 54.4(±.1) .37 .45 .41 .50 .34 .41
LoRA(0.35M) 46.7(±.4) 62.1(±.2) 55.3(±.2) .38 .44 .41 .46 .33 .39
GPT-2Large
Fine-Tune(774M) 41.7(±.5) 64.6(±.4) 54.2(±.4) .37 .46 .42 .56 .33 .43
Prefix(0.77M) 47.0(±.2) 64.2(±.4) 56.4(±.1) .39 .45 .42 .49 .33 .40
LoRA(0.77M) 48.4(±.3) 64.0(±.3) 57.0(±.1) .39 .45 .42 .45 .32 .38
Table11: GPT-2withdifferentadaptationmethodsonWebNLG.ThevariancesofMETandTER
arearound0.01varianceforalldifferentadaptionapproaches. “U”indicatesunseencategories,“S”
indicatesseencategories,and“A”indicatesallcategoriesinthetestsetofwebNLG.
(±0.3)varianceduetorandomseeds.
Method MNLI(m)-100 MNLI(m)-1k MNLI(m)-10k MNLI(m)-392K
GPT-3(Fine-Tune) 60.2 85.8 88.9 89.5
GPT-3(PrefixEmbed) 37.6 75.2 79.5 88.6
GPT-3(PrefixLayer) 48.3 82.5 85.9 89.6
GPT-3(LoRA) 63.8 85.6 89.2 91.7
Table12: ValidationaccuracyofdifferentmethodsonsubsetsofMNLIusingGPT-3175B.MNLI-n
describesasubsetwithntrainingexamples. Weevaluatewiththefullvalidationset. LoRAperforms
exhibitsfavorablesample-efficiencycomparedtoothermethods,includingfine-tuning.
ThetraininghyperparametersofdifferentadaptationapproachesonMNLI-narereportedinTable13.
WeuseasmallerlearningrateforPrefixLayerontheMNLI-100set,asthetraininglossdoesnot
decreasewithalargerlearningrate.
16

<!-- page 17 -->

Hyperparameters Adaptation MNLI-100 MNLI-1k MNLI-10K MNLI-392K
Optimizer - AdamW
WarmupTokens - 250,000
LRSchedule - Linear
BatchSize - 20 20 100 128
#Epoch - 40 40 4 2
FineTune 5.00E-6
PrefixEmbed 2.00E-04 2.00E-04 4.00E-04 5.00E-04
LearningRate
PrefixLayer 5.00E-05 5.00E-05 5.00E-05 1.00E-04
LoRA 2.00E-4
PrefixEmbedl 16 32 64 256
p
Adaptation- PrefixEmbedl 8
i
Specific PrefixTune l =l =8
p i
LoRA r =r =8
q v
Table13: ThehyperparametersusedfordifferentGPT-3adaptationmethodsonMNLI(m)-n.
F MeasuringSimilarityBetweenSubspaces
Inthispaperweusethemeasureφ(A,B,i,j)=ψ(Ui,Uj)= (cid:107)U A i(cid:62)UB(cid:107)2 F tomeasurethesubspace
A B min{i,j}
similarity between two column orthonormal matrices Ui ∈ Rd×i and Uj ∈ Rd×j, obtained by
A B
takingcolumnsoftheleftsingularmatricesofAandB. Wepointoutthatthissimilarityissimplya
reverseofthestandardProjectionMetricthatmeasuresdistancebetweensubspaces[? ].
Tobeconcrete, letthesingularvaluesofUi(cid:62)Uj tobeσ ,σ ,··· ,σ wherep = min{i,j}. We
A B 1 2 p
knowthattheProjectionMetric[? ] isdefinedas:
(cid:118)
(cid:117) p
d(Ui,Uj)= (cid:117) (cid:116)p− (cid:88) σ2 ∈[0, √ p]
A B i
i=1
whereoursimilarityisdefinedas:
(cid:80)p σ2 1(cid:16) (cid:17)
φ(A,B,i,j)=ψ(Ui,Uj)= i=1 i = 1−d(Ui,Uj)2
A B p p A B
ThissimilaritysatisfiesthatifUi andUj sharethesamecolumnspan,thenφ(A,B,i,j) = 1. If
A B
theyarecompletelyorthogonal,thenφ(A,B,i,j)=0. Otherwise,φ(A,B,i,j)∈(0,1).
G AdditionalExperimentsonLow-RankMatrices
Wepresentadditionalresultsfromourinvestigationintothelow-rankupdatematrices.
G.1 CorrelationbetweenLoRAModules
SeeFig.5andFig.6forhowtheresultspresentedinFig.3andFig.4generalizetootherlayers.
G.2 EffectofronGPT-2
Werepeatourexperimentontheeffectofr (Sec.6.2)inGPT-2. UsingtheE2ENLGChallenge
datasetasanexample,wereportthevalidationlossandtestmetricsachievedbydifferentchoices
ofraftertrainingfor26,000steps. WepresentourresultinTable14. TheoptimalrankforGPT-2
Mediumisbetween4and16dependingonthemetricused,whichissimilartothatforGPT-3175B.
Notethattherelationshipbetweenmodelsizeandtheoptimalrankforadaptationisstillanopen
question.
17

<!-- page 18 -->

1.0
0.8
0.6
0.4
0.2
0.0
1
reyaL
i
1
2
3
4
5
6
7
8
Wq Wv Wq Wv
23
reyaL
46
reyaL
i
i
1
2
3
4
5
6
7
8
1
2
3
4
5
6
7
8
1 6 21 81 32 92 53 04 64 25 85
j
69
reyaL
i
1
2
3
4
5
6
7
8
1 6 21 81 32 92 53 04 64 25 85
(Ar=8,Ar=64,i,j)
1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
j j j
Figure5: NormalizedsubspacesimilaritybetweenthecolumnvectorsofA andA forboth
r=8 r=64
∆W and∆W fromthe1st,32nd,64th,and96thlayersina96-layerTransformer.
q v
G.3 CorrelationbetweenW and∆W
SeeFig.7forthenormalizedsubspacesimilaritybetweenW and∆W withvaryingr.
Noteagainthat∆W doesnotcontainthetopsingulardirectionsofW,sincethesimilaritybetween
thetop4directionsin∆W andthetop-10%ofthoseinW barelyexceeds0.2. Thisgivesevidence
that∆W containsthose“task-specific”directionsthatareotherwisenotemphasizedinW.
Aninterestingnextquestiontoanswer,ishow“strong”doweneedtoamplifythosetask-specific
directions,inorderforthemodeladaptationtoworkwell?
G.4 AmplificationFactor
Onecannaturallyconsiderafeatureamplificationfactorastheratio (cid:107)∆W(cid:107)F ,whereU andV
(cid:107)U(cid:62)WV(cid:62)(cid:107)F
aretheleft-andright-singularmatricesoftheSVDdecompositionof∆W. (RecallUU(cid:62)WV(cid:62)V
givesthe“projection”ofW ontothesubspacespannedby∆W.)
Intuitively,when∆W mostlycontainstask-specificdirections,thisquantitymeasureshowmuchof
themareamplifiedby∆W. AsshowninSec.6.3,forr =4,thisamplificationfactorisaslargeas
20. Inotherwords,thereare(generallyspeaking)fourfeaturedirectionsineachlayer(outofthe
entirefeaturespacefromthepre-trainedmodelW),thatneedtobeamplifiedbyaverylargefactor
20, inordertoachieveourreportedaccuracyforthedownstreamspecifictask. And, oneshould
expectaverydifferentsetoffeaturedirectionstobeamplifiedforeachdifferentdownstreamtask.
18

<!-- page 19 -->

1
7
13
19 0.8 25 31 0.7
37
43 0.6
49
55 0.5
61
0.4
0.3
0.2
0.1
0.0
1 reyaL i
Wq Wv
23 reyaL
Wq Wv
1 6 11 61 12 62 13 63 14 64 15 65 16
1
7
13
19
25
31
37
43
49
55
61
j
46
reyaL i
1 6 11 61 12 62 13 63 14 64 15 65 16
j
1 6 11 61 12 62 13 63 14 64 15 65 16
j
69
reyaL
1 6 11 61 12 62 13 63 14 64 15 65 16
(Ar=64,A0r=64,i,j)
j
Figure6: NormalizedsubspacesimilaritybetweenthecolumnvectorsofA fromtworandomly
r=64
seeded runs, for both ∆W and ∆W from the 1st, 32nd, 64th, and 96th layers in a 96-layer
q v
Transformer.
Rankr val_loss BLEU NIST METEOR ROUGE_L CIDEr
1 1.23 68.72 8.7215 0.4565 0.7052 2.4329
2 1.21 69.17 8.7413 0.4590 0.7052 2.4639
4 1.18 70.38 8.8439 0.4689 0.7186 2.5349
8 1.17 69.57 8.7457 0.4636 0.7196 2.5196
16 1.16 69.61 8.7483 0.4629 0.7177 2.4985
32 1.16 69.33 8.7736 0.4642 0.7105 2.5255
64 1.16 69.24 8.7174 0.4651 0.7180 2.5070
128 1.16 68.73 8.6718 0.4628 0.7127 2.5030
256 1.16 68.92 8.6982 0.4629 0.7128 2.5012
512 1.16 68.78 8.6857 0.4637 0.7128 2.5025
1024 1.17 69.37 8.7495 0.4659 0.7149 2.5090
Table 14: Validation loss and test set metrics on E2E NLG Challenge achieved by LoRA with
differentrankrusingGPT-2Medium. UnlikeonGPT-3wherer =1sufficesformanytasks,here
theperformancepeaksatr = 16forvalidationlossandr = 4forBLEU,suggestingtheGPT-2
MediumhasasimilarintrinsicrankforadaptationcomparedtoGPT-3175B.Notethatsomeofour
hyperparametersaretunedonr =4,whichmatchestheparametercountofanotherbaseline,and
thusmightnotbeoptimalforotherchoicesofr.
One may notice, however, for r = 64, this amplification factor is only around 2, meaning that
mostdirectionslearnedin∆W withr = 64arenotbeingamplifiedbymuch. Thisshouldnotbe
surprising, andinfactgivesevidence(onceagain)thattheintrinsicrankneeded torepresentthe
“task-specificdirections”(thusformodeladaptation)islow. Incontrast,thosedirectionsintherank-4
versionof∆W (correspondingtor =4)areamplifiedbyamuchlargerfactor20.
19

<!-- page 20 -->

451
0.200
555
658 0.175
762
0.150
865
969 0.125
1072
0.100
1176
j
i
Wq Random
(Wq,Ar=4,i,j) (Wq,Ar=8,i,j) (Wq,Ar=64,i,j) (Wq,Arand,i,j)
j j j
Figure 7: Normalized subspace similarity between the singular directions of W and those of
q
∆W withvaryingr andarandombaseline. ∆W amplifiesdirectionsthatareimportantbutnot
q q
emphasizedinW. ∆W withalargerrtendstopickupmoredirectionsthatarealreadyemphasized
inW.
20
