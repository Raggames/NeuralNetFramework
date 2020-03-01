using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NeuralNetwork.Scripts.Data;
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
        [SerializeField] private NetData NetData;
        public string SaveNetDataFileName;
        
        [HideInInspector] public List<NeuralNet> NeuralNetworkInstances = new List<NeuralNet>();
        public NetData ActualBestDNA;
        private List<NetData> evaluateUpgradedsDNAOfEpoch = new List<NetData>();
        private List<NetData> keepAllInstancesDNAOfEpoch = new List<NetData>();
        public int Genetic_InstanceEndedCount;
        
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
            BackPropagate,
        }
        
        

       
        [SerializeField] public bool isNeuralNetTraining;
        [SerializeField] public bool isNeuralNetExecuting;
        
        [HideInInspector] public bool LoadFromBlueprint;
       

        
        [HideInInspector] public bool NewTraining = true;//check if instances is new or has already iterated
        [Header("Training Setup")] 
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
        
        [Header("Training : Genetic")]
        
        public double TrainingRate;//gradient d'update des weights
        public double MaxTrainingRate;
        public bool AdjustTrainingRateAutomatically;
        [Range(0.01f, 50f)] public float TrainingRateChangePurcentage; //if NeuralNetwork can't upgrade for n = "TrainingRateEvolution", decrease Training Rate by this value
        public int TrainingRateEvolution;
        private bool DNAShouldUpgrade;
        private bool ForceInstanceDNAReset; //if iteration don't get any upgrade avec n = TrainingRateEvolution epochs, reset on DNA data
        private int epochsWithoutDNAEvolutionCount;
        private List<double> previousIterationsCoefficients = new List<double>();
        [SerializeField] private double previousIterationsCoefficientAverage;
        
        public int EpochsCount;
        public float DelayBeforeRestart;
        [Header("Neural Network Evaluation")] 
        public int DNAVersion;
        public double[] TrainingBestResults;
        
        #endregion

        private void Start()
        {
            if (StartMode == EStartMode.NewTraining)
            {
                NewTraining = true;
            }
            else
            {
                NewTraining = false;
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.A))
            {
                if(runningMode == ERunningMode.Train) Train();
                if(runningMode == ERunningMode.Execute) Execute();
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
                NewTraining = NetData.NewTraining;
                ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
                ActualBestDNA.InstanceWeights = NetData.InstanceWeights;
                ActualBestDNA.InstanceBiases = NetData.InstanceWeights;
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
            netData.InstanceWeights.Clear();
            netData.InstanceBiases.Clear();
            netData.PerformanceCoefficient = 0;
            netData.PerformanceSolvers.Clear();
        }
        private void ForceSetManagerData()
        {
            NetData = LoadNetData(SaveNetDataFileName);
            Debug.Log("Loaded from Json");
            ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
            ActualBestDNA.InstanceWeights = NetData.InstanceWeights;
            ActualBestDNA.InstanceBiases = NetData.InstanceBiases;
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
            int _instanceID = 0;
            SetManagerData();
            for (int i = 0; i < batchSize; i++)
            {
                GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                NeuralNetworkInstances.Add(_instance);
                _instance.InitializeNeuralNetwork(eRunningMode, epochs, _instanceID, this, NetData);
               _instanceID++;
            }
            if (NewTraining)
            {
                TrainingBestResults = new double[NeuralNetworkInstances[0].OutputToExternal.Count];
            }
           
        }

        private void InitializeExecuting(GameObject networkPrefab, ERunningMode eRunningMode, int batchSize)
        {
            isNeuralNetTraining = false;
            isNeuralNetExecuting = true;
            int _instanceID = 0;
            ForceSetManagerData();
            for (int i = 0; i < batchSize; i++)
            {
                GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                NeuralNetworkInstances.Add(_instance);
                _instance.InitializeNeuralNetwork(eRunningMode, 1, _instanceID, this, NetData);
                _instanceID++;
            }
            
        }

        #endregion
        #region Evaluating

        private List<NetData> saveEpochNetEvaluate = new List<NetData>();
        private int SequenceCount;

        public void Genetic_OnInstanceCompare(NeuralNet instanceNetwork, double computedCoeff, List<NeuralNetworkPerformanceSolver> solvers, bool instancehasBestDna)
        {
            Genetic_InstanceEndedCount += 1;
            Genetic_SolverComparator(instanceNetwork, computedCoeff, solvers, instancehasBestDna);
            

        }

        private void Genetic_SolverComparator(NeuralNet instance, double computedCoeff, List<NeuralNetworkPerformanceSolver> solvers, bool instancehasBestDna)
        {
            bool hasComputedinstanceNetwork = false;
                NetData instDna = new NetData();
                instDna.InstanceWeights = new List<double>();
                instDna.InstanceBiases = new List<double>();
                instDna.InstanceWeights = instance.Ge(NetData, ELearningLogic.Genetic, TrainingRate)
                instDna.InstanceBiases = dna.Biases;
                
                instDna.PerformanceSolvers = new List<NeuralNetworkPerformanceSolver>();
                for (int i = 0; i < solvers.Count; i++)
                {
                    NeuralNetworkPerformanceSolver solver = new NeuralNetworkPerformanceSolver();
                    solver.ParameterName = solvers[i].ParameterName;
                    solver.ExpectedValue = solvers[i].ExpectedValue;
                    solver.ParameterType = solvers[i].ParameterType;
                    solver.EvaluationParameter = solvers[i].EvaluationParameter;
                    solver.EvaluationParameterWeight = solvers[i].EvaluationParameterWeight;
                    instDna.PerformanceSolvers.Add(solver);
                }
                // _instanceNetworkEvaluate.PerformanceSolvers = solvers;
                instDna.PerformanceCoefficient = computedCoeff;
                saveEpochNetEvaluate.Add(instDna); // list is filled with all item each epoch. 
                if (instancehasBestDna)
                {
                    Debug.Log("computed perf : " + computedCoeff);

                    evaluateUpgradedsDNAOfEpoch.Add(instDna);
                    DNAShouldUpgrade = true;
                }

                if (Genetic_InstanceEndedCount >= NeuralNetworkInstances.Count)
                {
                    if (NetData.HasData == false)
                    {
                        var best = ReturnBestCoefficientNetworkForThisIteration(evaluateUpgradedsDNAOfEpoch);
                        Debug.Log("BestPerf" + best.PerformanceCoefficient);
                        ActualBestDNA = best;
                        HandleAndDisplayResults(best.PerformanceSolvers);
                        SaveNetData(best.InstanceWeights, best.InstanceBiases, best.PerformanceSolvers,
                            best.PerformanceCoefficient);
                        DNAVersion = NetData.DNAVersion;
                        Debug.Log("Net Data was empty, actual iteration best DNA was saved.");
                        hasComputedinstanceNetwork = true;
                    }
                    if (DNAShouldUpgrade)
                    {
                        var best = ReturnBestCoefficientNetworkForThisIteration(evaluateUpgradedsDNAOfEpoch);
                        Debug.Log("BestPerf" + best.PerformanceCoefficient + " DNA Upgrade");
                        Genetic_ManageTrainingRateOnFeedback(DNAShouldUpgrade, best.PerformanceCoefficient,
                            ActualBestDNA.PerformanceCoefficient);
                        ActualBestDNA = best;
                        HandleAndDisplayResults(best.PerformanceSolvers);
                        SaveNetData(best.InstanceWeights, best.InstanceBiases, best.PerformanceSolvers,
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
                    if (Genetic_InstanceEndedCount >= NeuralNetworkInstances.Count && hasComputedinstanceNetwork)
                    {
                        Debug.Log("Starting Next Epoch : InstancesEndedCount = " + Genetic_InstanceEndedCount);
                        RestartEpochDelay();
                    }
                }
        }
        

        private void RestartEpochDelay()
        {
            StartCoroutine(WaitInstancesEvaluationBeforeRestart(DelayBeforeRestart));
        }
        
        IEnumerator WaitInstancesEvaluationBeforeRestart(float time)
        {
            float timer = 0;
            while (timer < time)
            {
                timer += Time.deltaTime;
                yield return null;
            }
            StartNextEpoch();
      
        }
        
        private NetData ReturnBestCoefficientNetworkForThisIteration(List<NetData> evaluateDnaForInstancesIteration)
        {
            evaluateDnaForInstancesIteration.Sort(delegate(NetData evaluate, NetData networkEvaluate)
                {
                    return evaluate.PerformanceCoefficient.CompareTo(networkEvaluate.PerformanceCoefficient);
                });
            
            return evaluateDnaForInstancesIteration[evaluateDnaForInstancesIteration.Count - 1];
        }

        public void ForceStartNextEpoch()
        {
            StartNextEpoch();
        }
        


        #endregion
        
        #region Restarting
        private void StartNextEpoch()
        {
            if (isNeuralNetTraining)
            {
                EpochsCount++;
                if(BackPropagation_SequenceIndex > SequenceCount)
                /////Case Genetic-------------------------------------------------------------------------------------
                if (Genetic_InstanceEndedCount >= NeuralNetworkInstances.Count)
                {
                    Genetic_RestartInstances(NeuralNetworkInstances);
                }
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
                
            }
            if (isNeuralNetExecuting)
            {
                NeuralNetworkInstances[0].Genetic_NeuralNetRestart(runningMode, NetData, DNAShouldUpgrade, ForceInstanceDNAReset);
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
            Debug.Log(Genetic_InstanceEndedCount + "instanceEndedcount after RestartInstance();");
            Genetic_InstanceEndedCount = 0;
            evaluateUpgradedsDNAOfEpoch.Clear();
            saveEpochNetEvaluate.Clear();
        }
        private void Genetic_ManageTrainingRateOnFeedback(bool dnaHasUpgrade, double bestCoeff = 0, double actualCoeff = 0)
        {
            if (AdjustTrainingRateAutomatically)
            {
                if (!dnaHasUpgrade)
                {
                    epochsWithoutDNAEvolutionCount++;
                    var best = ReturnBestCoefficientNetworkForThisIteration(saveEpochNetEvaluate);
                    previousIterationsCoefficients.Add(best.PerformanceCoefficient);
                    if (epochsWithoutDNAEvolutionCount > TrainingRateEvolution)
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
                    previousIterationsCoefficients.Clear();
                    previousIterationsCoefficientAverage = 0;
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
        
        #region DataManaging
        private void HandleAndDisplayResults(List<NeuralNetworkPerformanceSolver> solvers)
        {
            for (int i = 0; i < solvers.Count; i++)
            {
                TrainingBestResults[i] = solvers[i].EvaluationParameter;
            }
        }
        
        private void SaveNetData(List<double> instanceWeights, List<double> instanceBiases, List<NeuralNetworkPerformanceSolver> solvers, double notationCoefficient)
        {
            if (NewTraining)
            {
                NetData.StartTrainingRate = TrainingRate;
                NetData.NewTraining = false;
            }
            NewTraining = false;
            NetData.InstanceWeights = instanceWeights;
            NetData.InstanceBiases = instanceBiases;
            NetData.PerformanceSolvers = solvers;
            Debug.Log("changenetdata");
            NetData.PerformanceCoefficient = notationCoefficient;
            NetData.NeuralNetworkName = IANetworkName;
            NetData.NetworkTrainingRate = TrainingRate;
            NetData.DNAVersion++;
            NetData.HasData = true;
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