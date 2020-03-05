using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Scripts.NetToolbox;
using NeuralNetwork.Utils;
using UnityEngine;

namespace NeuralNetwork
{
    public class NeuralNetworkManager : MonoBehaviour
    {
        #region References
        [Header("Instanciation")]
        public GameObject NetWorkPrefab;
        public string IANetworkName = "Neural Network Exemple";
       
        [Header("Instances")]
        public string SaveNetDataFileName;
        [SerializeField] private NetData NetData;

        [HideInInspector] public List<NeuralNet> NeuralNetworkInstances = new List<NeuralNet>();
        public NetData ActualBestDNA;
        private List<NetData> evaluateUpgradedsDNAOfEpoch = new List<NetData>();
        private List<NetData> keepAllInstancesDNAOfEpoch = new List<NetData>();
        public int InstanceEndedCount;
        
        [Header("Networks Management")] 
        
        public ERunningMode runningMode;
        public enum ERunningMode
        {
            Train,
            Execute,
        }
        public ELearningLogic LearningLogic;
        public enum ELearningLogic
        {
            Genetic,
            BackPropagation,
        }
        [SerializeField] public bool isNeuralNetTraining;
        [SerializeField] public bool isNeuralNetExecuting;
        [HideInInspector] public bool LoadFromBlueprint;
        public bool NewExecution = true;//check if instances is new or has already iterated
        [Header("Training General Settings")] 
        public EStartMode StartMode;//should manager looks for an existing load and get it or no ?
        public enum EStartMode
        {
            NewTraining,
            LoadFromJson,
        }
        public ESaveMode SaveFile;
        public enum ESaveMode
        {
            Override,
            New,
        }
        public int Epochs;//Iterations of training
        public int TrainingBatchSize;//number of instances trained at the same time
        
        public NetworkTrainingStatistics NetworkTrainingStatistics = new NetworkTrainingStatistics();
        
        [Header("Training")]
        
        public double TrainingRate;//gradient d'update des weights
        public double InitialWeightsDelta; // should be something low for Data (0.01), Bigger for GameIA
        public int EpochsCount;

        [Header("BackPropagation Settings")] 
        public double WeightDecay = 0.0001f;
        public double Momentum = 0.01f;
        public float DataSplitPurcentage = 80;
        public double LossResultTrainingBreakPoint;
        public ELossFunction LossFunction;
        
        public enum ELossFunction
        {
            MeanSquarredError,
            MeanCrossEntropy,
        }     
        private List<NeuralNet> saveTrainingNeuralNetsForCompare = new List<NeuralNet>();

        [Header("Genetic Settings")]
        public double MaxTrainingRate;
        public bool AdjustTrainingRateAutomatically;
        [Range(0.01f, 50f)] public float TrainingRateChangePurcentage; //if NeuralNetwork can't upgrade for n = "TrainingRateEvolution", decrease Training Rate by this value
        public int EpochsBetweenTrainingRateChange;
        private bool DNAShouldUpgrade;
        private bool ForceInstanceDNAReset; //if iteration don't get any upgrade avec n = TrainingRateEvolution epochs, reset on DNA data
        private int epochsWithoutDNAEvolutionCount;
        private List<double> previousIterationsCoefficients = new List<double>();
        [SerializeField] private double previousIterationsCoefficientAverage;
        private List<NetData> epochNetDatas = new List<NetData>();
        private int SequenceCount;
        public float DelayBeforeRestart = 1;

        [Header("Training Sessions")] 
        public float EpochTimer;
        public double[] TrainingBestResults;
        private int DNAVersion;

        #endregion
        #region Execution
        private void Start()
        {
            NetworkTrainingStatistics.Name = IANetworkName + "_TrainingStatistics";
            
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.A))
            {
                if(runningMode == ERunningMode.Train) Train();
                if(runningMode == ERunningMode.Execute) Execute();
            }

            if (isNeuralNetTraining)
            {
                EpochTimer += Time.deltaTime;
            }
        }
        public void Train()
        {
            if(!isNeuralNetTraining) InitializeTraining(NetWorkPrefab, runningMode, TrainingBatchSize, Epochs);
        }
        public void Execute()
        {
            if(!isNeuralNetExecuting) InitializeExecuting(NetWorkPrefab, runningMode, TrainingBatchSize);
        }
        #endregion
        #region Initializing
        private void SetManagerData()
        {
            if (StartMode == EStartMode.NewTraining)
            {
                if (File.Exists(Application.dataPath + "/StreamingAssets/" + SaveNetDataFileName) && SaveFile == ESaveMode.Override)
                {
                    Debug.Log("An Existing save has been found, you should change the Save File Name if you don't wan't to override or set SaveMode to new file");
                }
                if (File.Exists(Application.dataPath + "/StreamingAssets/" + SaveNetDataFileName) && SaveFile == ESaveMode.New)
                {
                    Debug.Log("An Existing save has been found, setting File Name To New Name");
                    ResetNetData(NetData);
                    SaveNetDataFileName = IANetworkName + "_autoSave_" + NetData.autoSaveIncrement;
                    NetData.autoSaveIncrement++;
                    
                }

                else
                {
                    ResetNetData(NetData);
                }
                
            }
            if (StartMode == EStartMode.LoadFromJson)
            {
                NetData = LoadNetData(SaveNetDataFileName);
                Debug.Log("Loaded from Json");
                ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
                ActualBestDNA.InstanceWeights = NetData.InstanceWeights;
                ActualBestDNA.PerformanceCoefficient = NetData.PerformanceCoefficient;
                TrainingRate = NetData.NetworkTrainingRate;
                TrainingBestResults = new double[ActualBestDNA.PerformanceSolvers.Count];
                for (int j = 0; j < ActualBestDNA.PerformanceSolvers.Count; j++)
                {
                    TrainingBestResults[j] = ActualBestDNA.PerformanceSolvers[j].EvaluationParameter;
                }
            }
        }
        private void ResetNetData(NetData netData)
        {
            netData.HasData = false;
            netData.NeuralNetworkName = "";
            netData.DNAVersion = 0;
            netData.PerformanceCoefficient = 0;
            netData.PerformanceSolvers.Clear();
        }
        private void ForceSetManagerData()
        {
            NetData = LoadNetData(SaveNetDataFileName);
            Debug.Log("Loaded from Json");
            ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
            ActualBestDNA.InstanceWeights = NetData.InstanceWeights;
            ActualBestDNA.PerformanceCoefficient = NetData.PerformanceCoefficient;
            TrainingRate = NetData.NetworkTrainingRate;
            TrainingBestResults = new double[ActualBestDNA.PerformanceSolvers.Count];
            for (int j = 0; j < ActualBestDNA.PerformanceSolvers.Count; j++)
            {
                TrainingBestResults[j] = ActualBestDNA.PerformanceSolvers[j].EvaluationParameter;
            }
        }
        private void InitializeTraining(GameObject networkPrefab, ERunningMode eRunningMode, int batchSize, int epochs)
        {
            isNeuralNetTraining = true;
            isNeuralNetExecuting = false;
            
            if (NeuralNetworkInstances.Count == 0)
            {
                int _instanceID = 0;
                SetManagerData();
                for (int i = 0; i < batchSize; i++)
                {
                    GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                    NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                    NeuralNetworkInstances.Add(_instance);
                    _instance.InitializeTraining(eRunningMode,this,  epochs, _instanceID, NetData);
                    _instanceID++;
                }
                if (NewExecution)
                {
                    TrainingBestResults = new double[NeuralNetworkInstances[0].Controller.EvaluationParameters.Count];
                }
            }
            else
            {
                EpochsCount = 0;
                InstanceEndedCount = 0;
                foreach (var Instance in NeuralNetworkInstances)
                {
                   Instance.InitializeTraining(eRunningMode,this,  epochs, Instance.InstanceID, NetData); 
                }
            }
            
            
        }
        private void InitializeExecuting(GameObject networkPrefab, ERunningMode eRunningMode, int batchSize)
        {
            isNeuralNetTraining = false;
            isNeuralNetExecuting = true;
            int _instanceID = 0;
            ForceSetManagerData();
            if (NeuralNetworkInstances.Count == 0)
            {
                for (int i = 0; i < batchSize; i++)
                {
                    GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                    NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                    NeuralNetworkInstances.Add(_instance);
                    _instance.InitializeTraining(eRunningMode, this, 1, _instanceID, NetData);
                    _instanceID++;
                }
            }
            else
            {
                foreach (var Instance in NeuralNetworkInstances)
                {
                    Instance.InitializeTraining(eRunningMode,this,  1, Instance.InstanceID, NetData); 
                }
            }
            
        }
        #endregion
        #region Genetic_TrainingAlgorithm
      
        public void Genetic_OnInstanceCompare(NeuralNet instanceNetwork, double computedCoeff, List<NetLossParameter> solvers, bool instancehasBestDna)
        {
            Genetic_LossResultsComparator(instanceNetwork, computedCoeff, solvers, instancehasBestDna);

        }
        private void Genetic_LossResultsComparator(NeuralNet instance, double computedCoeff, List<NetLossParameter> solvers, bool instancehasBestDna)
        {
            InstanceEndedCount += 1;
            bool hasComputedinstanceNetwork = false;
                NetData instDna = new NetData();
                instDna.InstanceWeights = new double[instance.WeightsNumber];
                instDna.InstanceWeights = instance._NetData.InstanceWeights;
                instDna.PerformanceSolvers = new List<NetLossParameter>();
                for (int i = 0; i < solvers.Count; i++)
                {
                    NetLossParameter solver = new NetLossParameter();
                    solver.ParameterName = solvers[i].ParameterName;
                    solver.ExpectedValue = solvers[i].ExpectedValue;
                    solver.ParameterType = solvers[i].ParameterType;
                    solver.EvaluationParameter = solvers[i].EvaluationParameter;
                    solver.EvaluationParameterWeight = solvers[i].EvaluationParameterWeight;
                    instDna.PerformanceSolvers.Add(solver);
                }
                // _instanceNetworkEvaluate.PerformanceSolvers = solvers;
                instDna.PerformanceCoefficient = computedCoeff;
                epochNetDatas.Add(instDna); // list is filled with all item each epoch. 
                if (instancehasBestDna)
                {
                    Debug.Log("computed perf : " + computedCoeff);

                    evaluateUpgradedsDNAOfEpoch.Add(instDna);
                    DNAShouldUpgrade = true;
                }

                if (InstanceEndedCount >= NeuralNetworkInstances.Count)
                {
                    if (NetData.HasData == false)
                    {
                        var best = Genetic_ComputeBestPerformanceIndex(evaluateUpgradedsDNAOfEpoch);
                        Debug.Log("BestPerf" + best.PerformanceCoefficient);
                        ActualBestDNA = best;
                        HandleAndDisplayResults(best.PerformanceSolvers);
                        SaveNetData(best.InstanceWeights, instance, best.PerformanceSolvers,
                            best.PerformanceCoefficient);
                        DNAVersion = NetData.DNAVersion;
                        Debug.Log("Net Data was empty, actual iteration best DNA was saved.");
                        hasComputedinstanceNetwork = true;
                    }
                    if (DNAShouldUpgrade)
                    {
                        var best = Genetic_ComputeBestPerformanceIndex(evaluateUpgradedsDNAOfEpoch);
                        Debug.Log("BestPerf" + best.PerformanceCoefficient + " DNA Upgrade");
                        Genetic_ManageTrainingRateOnFeedback(DNAShouldUpgrade, best.PerformanceCoefficient,
                            ActualBestDNA.PerformanceCoefficient);
                        ActualBestDNA = best;
                        HandleAndDisplayResults(best.PerformanceSolvers);
                        SaveNetData(best.InstanceWeights, instance, best.PerformanceSolvers,
                            best.PerformanceCoefficient);
                        DNAVersion = NetData.DNAVersion;
                        hasComputedinstanceNetwork = true;
                    }
                    if (!DNAShouldUpgrade)
                    {
                       Genetic_ManageTrainingRateOnFeedback(DNAShouldUpgrade, 0f,
                            ActualBestDNA.PerformanceCoefficient);
                        hasComputedinstanceNetwork = true;
                    }

                    //InstancesEndedCount++;
                    if (InstanceEndedCount >= NeuralNetworkInstances.Count && hasComputedinstanceNetwork)
                    {
                        Debug.Log("Starting Next Epoch : InstancesEndedCount = " + InstanceEndedCount);
                        Genetic_RestartEpochDelay();
                    }
                }
        }
        private void Genetic_RestartEpochDelay()
        {
            StartCoroutine(Genetic_WaitDelayBeforeNewEpoch(DelayBeforeRestart));
        }
        IEnumerator Genetic_WaitDelayBeforeNewEpoch(float time)
        {
            float timer = 0;
            while (timer < time)
            {
                timer += Time.deltaTime;
                yield return null;
            }
            Genetic_StartNextEpoch();
      
        }
        private NetData Genetic_ComputeBestPerformanceIndex(List<NetData> evaluateDnaForInstancesIteration)
        {
            evaluateDnaForInstancesIteration.Sort(delegate(NetData evaluate, NetData networkEvaluate)
                {
                    return evaluate.PerformanceCoefficient.CompareTo(networkEvaluate.PerformanceCoefficient);
                });
            
            return evaluateDnaForInstancesIteration[evaluateDnaForInstancesIteration.Count - 1];
        }
        public void Genetic_ForceStartNextEpoch()
        {
            Genetic_StartNextEpoch();
        }
        private void Genetic_StartNextEpoch()
        {
            if (isNeuralNetTraining)
            {
                EpochsCount++;
                EpochTimer = 0;
                // Epochs Are Done------------------------------------------------------------------------------------
                if (EpochsCount >= Epochs)
                {
                    isNeuralNetTraining = false;
                    foreach (var instanceNet in NeuralNetworkInstances)
                    {
                        instanceNet.IsTraining = false;
                        instanceNet.IsExecuting = false;
                        EpochsCount = 0;
                    }
                    Debug.Log("ENDING TRAINING EPOCHS");
                }
                /////Case Genetic-------------------------------------------------------------------------------------
                if (InstanceEndedCount >= NeuralNetworkInstances.Count)
                {
                    Genetic_RestartInstances(NeuralNetworkInstances);
                }
            }
            if (isNeuralNetExecuting)
            {
                    Genetic_RestartInstances(NeuralNetworkInstances);
            }
        }
        private void Genetic_RestartInstances(List<NeuralNet> neuralNets)
        {
            foreach (var netInstance in neuralNets)
            {
                netInstance.Genetic_NeuralNetRestart(runningMode, NetData, DNAShouldUpgrade, ForceInstanceDNAReset);
            }
            ForceInstanceDNAReset = false;
            DNAShouldUpgrade = false;
            Debug.Log(InstanceEndedCount + "instanceEndedcount after RestartInstance();");
            InstanceEndedCount = 0;
            evaluateUpgradedsDNAOfEpoch.Clear();
            epochNetDatas.Clear();
        }
        private void Genetic_ManageTrainingRateOnFeedback(bool dnaHasUpgrade, double bestCoeff = 0, double actualCoeff = 0)
        {
            if (AdjustTrainingRateAutomatically)
            {
                if (!dnaHasUpgrade)
                {
                    epochsWithoutDNAEvolutionCount++;
                    var best = Genetic_ComputeBestPerformanceIndex(epochNetDatas);
                    previousIterationsCoefficients.Add(best.PerformanceCoefficient);
                    // Statistic========================================================================================
                    SaveTrainingStatData(best.PerformanceCoefficient);
                    //==================================================================================================
                    if (epochsWithoutDNAEvolutionCount > EpochsBetweenTrainingRateChange)
                    {
                        ForceInstanceDNAReset = true;
                        double averageCoefficient = 0;
                        double actualTrainingRate = TrainingRate;
                        for (int i = 0; i < previousIterationsCoefficients.Count; i++)
                        {
                            averageCoefficient += previousIterationsCoefficients[i];
                        }
                        averageCoefficient /= previousIterationsCoefficients.Count;
                        previousIterationsCoefficientAverage = averageCoefficient;
                        double downgradeDelta = 0;
                        if (averageCoefficient > 0)
                        {
                            downgradeDelta = ((actualCoeff - averageCoefficient)/averageCoefficient)*TrainingRateChangePurcentage/100;
                        }
                        else
                        {
                            downgradeDelta = actualCoeff*TrainingRateChangePurcentage/100;
                        }
                        actualTrainingRate += downgradeDelta;
                        Debug.Log("Training Rate Increased from " + TrainingRate + " to "+ actualTrainingRate);
                        TrainingRate = actualTrainingRate;
                        TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                        epochsWithoutDNAEvolutionCount = 0;
                        
                    }
                }
                if (dnaHasUpgrade)
                {
                    // Statistic========================================================================================
                    SaveTrainingStatData(bestCoeff);
                    //==================================================================================================
                    previousIterationsCoefficients.Clear();
                    previousIterationsCoefficients.Add(bestCoeff);
                    previousIterationsCoefficientAverage = bestCoeff;
                    double actualTrainingRate = TrainingRate;
                    double upgradeDelta = Mathf.Abs((float)bestCoeff - (float)actualCoeff) / bestCoeff * TrainingRateChangePurcentage / 100;
                    actualTrainingRate -= upgradeDelta;
                    Debug.Log("Training Rate Decreased from " + TrainingRate + " to " + actualTrainingRate);
                    TrainingRate = actualTrainingRate;
                    TrainingRate = Mathf.Clamp((float) TrainingRate, 0.000001f, (float) MaxTrainingRate);
                }
            }

        }
    
        #endregion
        #region BackPropagation_Training

        public void BackPropagation_OnEpochDone(NeuralNet instance)
        {
            InstanceEndedCount += 1;
            if (InstanceEndedCount >= NeuralNetworkInstances.Count)
            {
               // End Training
                EpochsCount++;
                if (EpochsCount < Epochs)
                {
                    StartCoroutine(BackPropagation_WaitDelayBeforeNewEpoch(DelayBeforeRestart));
                }
                
                if (EpochsCount == Epochs && isNeuralNetTraining)
                {
                    BackPropagation_OnTrainingDone(instance);
                }
               
            }
        }

        public void BackPropagation_OnTrainingDone(NeuralNet instance)
        {
            
            NetData instDna = new NetData();
            instDna.InstanceWeights = new double[instance.WeightsNumber];
            instDna.InstanceWeights = instance._NetData.InstanceWeights;
            // _instanceNetworkEvaluate.PerformanceSolvers = solvers;
            instance.GetAccuracyOnTestData();
            instance.GetErrorOnTestData();
            instDna.PerformanceCoefficient = instance._NetData.PerformanceCoefficient;
            instDna.DataPrediction_Accuracy = instance._NetData.DataPrediction_Accuracy;
            SaveTrainingStatData(instance._NetData.PerformanceCoefficient, instance._NetData.DataPrediction_Accuracy);
            saveTrainingNeuralNetsForCompare.Add(instance); // list is filled with all item each epoch. 
            var bestInstance = BackPropagation_CompareAccuracyResults(saveTrainingNeuralNetsForCompare);
            Debug.Log("Best Accuracy Result for training is : " + bestInstance._NetData.DataPrediction_Accuracy);
            Debug.Log("Best Instance is : " + bestInstance.gameObject);
            SaveNetData(bestInstance._NetData.InstanceWeights, bestInstance, bestInstance.Controller.EvaluationParameters, bestInstance._NetData.PerformanceCoefficient);
            isNeuralNetTraining = false;
            EpochsCount = 0;
            InstanceEndedCount = 0;
            
            Debug.Log("TrainingEnd");

        }
        private void BackPropagation_StartNextEpoch()
        {
            foreach (var inst in NeuralNetworkInstances)
            {
                inst.ComputeAndTrain(TrainingRate, Momentum, WeightDecay);
            }
        }
        IEnumerator BackPropagation_WaitDelayBeforeNewEpoch(float time)
        {
            float timer = 0;
            while (timer < time)
            {
                timer += Time.deltaTime;
                yield return null;
            }
            BackPropagation_StartNextEpoch();
      
        }
        private NeuralNet BackPropagation_CompareAccuracyResults(List<NeuralNet> neuralNets)
        {
            saveTrainingNeuralNetsForCompare.Sort(delegate(NeuralNet aData, NeuralNet bData)
                {
                    return aData._NetData.PerformanceCoefficient.CompareTo(bData._NetData.PerformanceCoefficient);
                });
            return saveTrainingNeuralNetsForCompare[0];
        }
        
        #endregion
        
        #region DataManaging
        private void HandleAndDisplayResults(List<NetLossParameter> solvers)
        {
            for (int i = 0; i < solvers.Count; i++)
            {
                TrainingBestResults[i] = solvers[i].EvaluationParameter;
            }
        }

        public void SaveTrainingStatData(double performanceCoefficient, double accuracy = 0)
        {
            NetworkTrainingStatistics.SetStatEntry(EpochsCount, performanceCoefficient, TrainingRate, accuracy);
            TrainingStatsData statData = new TrainingStatsData();
            statData.Name = NetworkTrainingStatistics.Name;
            statData.TrainingSessionStatistics = NetworkTrainingStatistics.TrainingSessionStatistics;
            NeuralNetworkSerializer.GenericSave(statData, NetworkTrainingStatistics.Name);
        }
        private void SaveNetData(double[] instanceWeights, NeuralNet instance, List<NetLossParameter> solvers, double notationCoefficient)
        {
            if (NewExecution)
            {
                NetData.StartTrainingRate = TrainingRate;
                NetData.NewTraining = false;
            }
            NewExecution = false;
            NetData.InstanceWeights = new double[instance.WeightsNumber];
            NetData.InstanceWeights = instanceWeights;
            NetData.PerformanceSolvers = solvers;
            Debug.Log("changenetdata");
            NetData.PerformanceCoefficient = notationCoefficient;
            NetData.NeuralNetworkName = IANetworkName;
            NetData.NetworkTrainingRate = TrainingRate;
            NetData.DNAVersion++;
            NetData.HasData = true;
            ActualBestDNA = NetData;
            // Serialize
            NeuralNetworkSerializer.Save(NetData, SaveNetDataFileName);
        }

        private NetData LoadNetData(string fileName)
        {
            Scripts.Data.NetData loadedData = new NetData();
            loadedData = NeuralNetworkSerializer.Load(loadedData, fileName);
            return loadedData;
        }
      
        #endregion
    }
}